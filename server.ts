import "dotenv/config";
import express, { type Request, type Response } from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function getAiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY is not configured on the server.");
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

function safeJsonParse<T = Record<string, unknown>>(raw: string | undefined): T {
  const cleaned = (raw ?? "{}")
    .trim()
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();
  return JSON.parse(cleaned || "{}") as T;
}

async function withRetry<T>(fn: () => Promise<T>, maxAttempts = 2): Promise<T> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error: unknown) {
      lastError = error;
      const status =
        typeof error === "object" && error !== null && "status" in error
          ? Number((error as { status?: unknown }).status)
          : undefined;
      const isRetryable = status !== undefined && status >= 500;
      if (!isRetryable || attempt === maxAttempts) {
        throw error;
      }
      const delayMs = 400 * 2 ** (attempt - 1);
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }
  throw lastError;
}

function getErrorStatusAndMessage(error: unknown): { status: number; message: string } {
  const errObj = typeof error === "object" && error !== null ? (error as Record<string, unknown>) : {};
  let status = typeof errObj.status === "number" && errObj.status >= 400 && errObj.status < 600
    ? errObj.status
    : 500;
  let rawMessage =
    error instanceof Error
      ? error.message
      : typeof errObj.message === "string"
        ? errObj.message
        : "Internal server error";

  let apiStatusCode = "";
  if (rawMessage.trim().startsWith("{")) {
    try {
      const parsed = JSON.parse(rawMessage) as {
        error?: { code?: number; message?: string; status?: string };
      };
      if (parsed.error) {
        if (typeof parsed.error.code === "number" && parsed.error.code >= 400 && parsed.error.code < 600) {
          status = parsed.error.code;
        }
        if (parsed.error.status) {
          apiStatusCode = parsed.error.status;
        }
        if (parsed.error.message) {
          rawMessage = parsed.error.message;
        }
      }
    } catch {
      // Keep rawMessage if not valid JSON
    }
  }

  if (status === 429 || apiStatusCode === "RESOURCE_EXHAUSTED") {
    return {
      status: 429,
      message:
        "Gemini API quota exceeded. Please try again later or select a billing-enabled API key in Settings > Secrets.",
    };
  }

  if (status === 403 || apiStatusCode === "PERMISSION_DENIED" || apiStatusCode === "API_KEY_INVALID") {
    return {
      status,
      message: "Invalid or unauthorized Gemini API key. Please check your key in Settings > Secrets.",
    };
  }

  return { status, message: rawMessage };
}

interface MarketDataPayload {
  prices: Record<string, unknown> | null;
  markets: Array<{ title: string; uri: string }>;
}

const MARKET_CACHE_TTL_MS = 30 * 60 * 1000;
const marketDataCache = new Map<string, { data: MarketDataPayload; expiresAt: number }>();

async function startServer() {
  const app = express();
  const PORT = Number(process.env.PORT) || 3000;

  app.use(express.json({ limit: "15mb" }));

  app.post("/api/gemini/analyze-animal", async (req: Request, res: Response) => {
    try {
      const { imageBase64, mimeType } = req.body as {
        imageBase64?: string;
        mimeType?: string;
      };

      if (!imageBase64 || typeof imageBase64 !== "string") {
        res.status(400).json({ error: "Missing imageBase64 in request body." });
        return;
      }

      const ai = getAiClient();
      const response = await withRetry(() =>
        ai.models.generateContent({
          model: "gemini-4-hydrogen",
          contents: {
            parts: [
              {
                inlineData: {
                  data: imageBase64,
                  mimeType: mimeType || "image/jpeg",
                },
              },
              {
                text: `Analyze this animal. Provide its breed/type, health estimation, and approximate market value in Tajikistan and Uzbekistan.
Format response as JSON with keys: "breed" (string), "price" (string with range in TJS/UZS) and "info" (brief description).`,
              },
            ],
          },
          config: {
            responseMimeType: "application/json",
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                breed: { type: Type.STRING },
                price: { type: Type.STRING },
                info: { type: Type.STRING },
              },
              required: ["breed", "price", "info"],
            },
          },
        })
      );

      const result = safeJsonParse(response.text);
      res.json(result);
    } catch (error) {
      const { status, message } = getErrorStatusAndMessage(error);
      if (status === 429) {
        console.warn("Rate limit / quota reached in /api/gemini/analyze-animal:", message);
      } else {
        console.error("Error in /api/gemini/analyze-animal:", message);
      }
      res.status(status).json({ error: message });
    }
  });

  app.post("/api/gemini/voice-assistant", async (req: Request, res: Response) => {
    try {
      const { transcript, audio, mimeType, lang, listings } = req.body as {
        transcript?: string;
        audio?: string;
        mimeType?: string;
        lang?: "ru" | "uz" | "tg";
        listings?: Array<{
          title?: string;
          price?: number;
          currency?: string;
          location?: string;
        }>;
      };

      if ((!transcript || typeof transcript !== "string") && !audio) {
        res.status(400).json({ error: "Missing transcript or audio in request body." });
        return;
      }

      const languageName = lang === "ru" ? "Russian" : lang === "uz" ? "Uzbek" : "Tajik";
      const safeListings = Array.isArray(listings) ? listings.slice(0, 10) : [];
      const listingSummary = safeListings
        .map((l) => `${l.title || ""} for ${l.price ?? ""} ${l.currency || ""} in ${l.location || ""}`)
        .join(", ");

      // VOICE_AUDIO_FIX: accept recorded audio, transcribe + answer in one call
      if (audio) {
        const aiAudio = getAiClient();
        const audioResp = await withRetry(() =>
          aiAudio.models.generateContent({
            model: "gemini-4-hydrogen",
            contents: {
              parts: [
                { inlineData: { data: audio, mimeType: mimeType || "audio/wav" } },
                {
                  text: `The user spoke the audio above to an app for buying/selling livestock (cows, sheep, horses, etc.) in Tajikistan and Uzbekistan.
Available listings: ${listingSummary}.
1) Transcribe what the user said. 2) Give a helpful, very concise answer (max 2 sentences) in ${languageName}.
If the audio is silent or unclear, set "answer" to a short polite request to repeat, in ${languageName}.
Reply ONLY as JSON: {"transcript": "...", "answer": "..."}`,
                },
              ],
            },
            config: { responseMimeType: "application/json" },
          })
        );
        let parsed: { transcript?: string; answer?: string } = {};
        try {
          parsed = safeJsonParse(audioResp.text);
        } catch {
          parsed = { answer: audioResp.text };
        }
        res.json({
          text: parsed.answer || "Sorry, I couldn't process that.",
          transcript: parsed.transcript || "",
        });
        return;
      }

      const ai = getAiClient();
      const response = await withRetry(() =>
        ai.models.generateContent({
          model: "gemini-4-hydrogen",
          contents: `The user said: "${transcript}". 
This is an app for buying/selling livestock (cows, sheep, horses, etc.) in Tajikistan and Uzbekistan.
Available listings: ${listingSummary}.
Provide a helpful, short response in ${languageName}.
If they are looking for something, tell them what's available. 
Keep it very concise (max 2 sentences).`,
        })
      );

      res.json({
        text: response.text || "Sorry, I couldn't process that.",
      });
    } catch (error) {
      const { status, message } = getErrorStatusAndMessage(error);
      if (status === 429) {
        console.warn("Rate limit / quota reached in /api/gemini/voice-assistant:", message);
      } else {
        console.error("Error in /api/gemini/voice-assistant:", message);
      }
      res.status(status).json({ error: message });
    }
  });

  app.post("/api/gemini/tts", async (req: Request, res: Response) => {
    try {
      const { text } = req.body as { text?: string };

      if (!text || typeof text !== "string") {
        res.status(400).json({ error: "Missing text in request body." });
        return;
      }

      const ai = getAiClient();
      const response = await withRetry(() =>
        ai.models.generateContent({
          model: "gemini-3.8-flash-lite-tts",
          contents: [{ role: "user", parts: [{ text }] }],
          config: {
            responseModalities: ["AUDIO"],
            speechConfig: {
              voiceConfig: {
                prebuiltVoiceConfig: { voiceName: "Kore" },
              },
            },
          },
        })
      );

      const inlineData = response.candidates?.[0]?.content?.parts?.[0]?.inlineData;
      if (!inlineData?.data) {
        res.status(502).json({ error: "No audio data returned from TTS model." });
        return;
      }

      res.json({
        audio: inlineData.data,
        mimeType: inlineData.mimeType || "audio/wav",
      });
    } catch (error) {
      const { status, message } = getErrorStatusAndMessage(error);
      if (status === 429) {
        console.warn("Rate limit / quota reached in /api/gemini/tts:", message);
      } else {
        console.error("Error in /api/gemini/tts:", message);
      }
      res.status(status).json({ error: message });
    }
  });

  app.post("/api/gemini/market-data", async (req: Request, res: Response) => {
    try {
      const { location, forceRefresh } = req.body as {
        location?: { latitude?: number; longitude?: number };
        forceRefresh?: boolean;
      };

      const validLocation = {
        latitude:
          typeof location?.latitude === "number" && Number.isFinite(location.latitude)
            ? location.latitude
            : 38.5358,
        longitude:
          typeof location?.longitude === "number" && Number.isFinite(location.longitude)
            ? location.longitude
            : 68.7791,
      };

      const cacheKey = `${validLocation.latitude.toFixed(1)},${validLocation.longitude.toFixed(1)}`;
      const cached = marketDataCache.get(cacheKey);
      if (!forceRefresh && cached && cached.expiresAt > Date.now()) {
        res.json(cached.data);
        return;
      }

      const ai = getAiClient();

      let prices: Record<string, unknown> | null = null;
      let markets: Array<{ title: string; uri: string }> = [];
      let firstError: unknown = null;

      try {
        const priceResponse = await withRetry(() =>
          ai.models.generateContent({
            model: "gemini-4-hydrogen",
            contents: `Provide current average market prices for beef, mutton, wheat, and corn in Tajikistan and Uzbekistan. 
Format as JSON with keys: beef_tj, mutton_tj, wheat_tj, corn_tj, beef_uz, mutton_uz, wheat_uz, corn_uz. 
Values should be numbers in local currency (TJS/UZS).`,
            config: {
              tools: [{ googleSearch: {} }],
              responseMimeType: "application/json",
              responseSchema: {
                type: Type.OBJECT,
                properties: {
                  beef_tj: { type: Type.NUMBER },
                  mutton_tj: { type: Type.NUMBER },
                  wheat_tj: { type: Type.NUMBER },
                  corn_tj: { type: Type.NUMBER },
                  beef_uz: { type: Type.NUMBER },
                  mutton_uz: { type: Type.NUMBER },
                  wheat_uz: { type: Type.NUMBER },
                  corn_uz: { type: Type.NUMBER },
                },
              },
            },
          })
        );
        prices = safeJsonParse(priceResponse.text);
      } catch (err) {
        firstError = err;
      }

      try {
        const marketResponse = await withRetry(() =>
          ai.models.generateContent({
            model: "gemini-4-hydrogen",
            contents:
              "Find animal markets (cattle markets) in Tajikistan and Uzbekistan. Tell me where and when they take place (days of the week).",
            config: {
              tools: [{ googleMaps: {} }],
              toolConfig: {
                retrievalConfig: {
                  latLng: validLocation,
                },
              },
            },
          })
        );
        const groundingChunks =
          marketResponse.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
        markets = groundingChunks
          .filter((chunk) => Boolean(chunk.maps?.uri))
          .map((chunk) => ({
            title: chunk.maps?.title || "Animal Market",
            uri: chunk.maps?.uri || "",
          }));
      } catch (err) {
        if (!firstError) {
          firstError = err;
        }
      }

      if (prices === null && markets.length === 0 && firstError) {
        if (cached) {
          const { message } = getErrorStatusAndMessage(firstError);
          res.json({ ...cached.data, warning: message });
          return;
        }
        throw firstError;
      }

      const payload: MarketDataPayload = { prices, markets };
      if (prices !== null && markets.length > 0) {
        marketDataCache.set(cacheKey, {
          data: payload,
          expiresAt: Date.now() + MARKET_CACHE_TTL_MS,
        });
      }

      if (firstError) {
        const { message } = getErrorStatusAndMessage(firstError);
        res.json({ ...payload, warning: message });
        return;
      }

      res.json(payload);
    } catch (error) {
      const { status, message } = getErrorStatusAndMessage(error);
      if (status === 429) {
        console.warn("Rate limit / quota reached in /api/gemini/market-data:", message);
      } else {
        console.error("Error in /api/gemini/market-data:", message);
      }
      res.status(status).json({ error: message });
    }
  });

  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, "dist");
    app.use(express.static(distPath));
    app.get("*", (_req: Request, res: Response) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
