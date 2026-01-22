"use client";

import { useState, KeyboardEvent, useRef, useEffect } from "react";
import dynamic from "next/dynamic";
import {
  Send,
  UtensilsCrossed,
  MapPin,
  Clock,
  ChefHat,
  Sparkles,
  ArrowRight,
  BookOpen
} from "lucide-react";
import ReactMarkdown from "react-markdown";

// 1. Dynamic Import for Map (Avoids SSR issues)
const MapPanel = dynamic(() => import("@/components/MapPanel"), {
  ssr: false,
  loading: () => <div className="w-full h-full bg-[#151515] animate-pulse flex items-center justify-center text-gray-800">Loading Map...</div>
});

// Updated to match the backend Generator's response structure
type Source = {
  restaurant: string;
  address: string;
  city?: string;
  state?: string;
  text?: string;
  lat?: number;
  lon?: number;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[]; // Attached sources to the message state
};

export default function Home() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);

  // These sources drive the MapPanel on the right
  const [activeSources, setActiveSources] = useState<Source[]>([]);
  const [busy, setBusy] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  async function send(queryOverride?: string) {
    const textToSend = queryOverride || input;
    if (!textToSend.trim()) return;

    // 1. Add User Message
    setMessages((prev) => [...prev, { role: "user", content: textToSend }]);
    setBusy(true);
    setInput("");
    setActiveSources([]); // Clear map for new search

    try {
      // NOTE: Ensure you have a Next.js API route at /api/chat that proxies to your Python backend
      // OR point this directly to "http://localhost:8001/generate" if CORS allows.
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: textToSend,
          city: null, // We let the backend NLP extract the city from the query string
          top_k: 12,
        }),
      });

      if (!res.ok || !res.body) throw new Error(`Server error: ${res.status}`);

      // 2. Add Placeholder Assistant Message
      let assistantText = "";
      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Handle potentially multiple JSON objects in one chunk
        const lines = buffer.split("\n");
        // Keep the last segment in buffer if it's incomplete
        buffer = lines.pop()!;

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const json = JSON.parse(line);

            // --- Handle Sources (First event from backend) ---
            if (json.type === "sources") {
              const fetchedSources: Source[] = json.data;
              setActiveSources(fetchedSources);

              // Attach sources to the current assistant message immediately
              setMessages((prev) => {
                const copy = [...prev];
                const lastIdx = copy.length - 1;
                if (copy[lastIdx].role === "assistant") {
                  copy[lastIdx] = { ...copy[lastIdx], sources: fetchedSources };
                }
                return copy;
              });
            }
            // --- Handle Tokens (Streaming text) ---
            else if (json.type === "token") {
              assistantText += json.data;
              setMessages((prev) => {
                const copy = [...prev];
                const lastIdx = copy.length - 1;
                if (copy[lastIdx].role === "assistant") {
                  copy[lastIdx] = { ...copy[lastIdx], content: assistantText };
                }
                return copy;
              });
            }
            // --- Handle Errors ---
            else if (json.type === "error") {
              console.error("Backend Error:", json.data);
              assistantText += `\n\n*[System Error: ${json.data}]*`;
            }
          } catch (e) {
            console.error("JSON Parse Error", e);
          }
        }
      }
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "❌ Error: Could not connect to the food guide API." },
      ]);
    } finally {
      setBusy(false);
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  // --- RENDER HELPERS ---

  // 1. Hero State (Empty Chat)
  const renderHero = () => (
    <div className="flex-1 flex flex-col items-center justify-center p-6 text-center animate-in fade-in zoom-in-95 duration-500">
      <div className="w-24 h-24 rounded-full bg-[#1a1a1a] border border-[#333] flex items-center justify-center mb-8 shadow-2xl">
        <UtensilsCrossed className="w-10 h-10 text-orange-500" strokeWidth={1.5} />
      </div>
      <h2 className="text-4xl md:text-5xl font-serif font-medium text-white mb-4 tracking-tight">
        Hungry? Let's find a spot.
      </h2>
      <p className="text-gray-400 text-lg max-w-lg font-light mb-12">
        Ask me anything about local restaurants, and I'll help you discover the perfect place to eat.
      </p>
      <div className="flex flex-wrap justify-center gap-3">
        {[
          { label: "Best tacos in town", icon: UtensilsCrossed },
          { label: "Romantic Italian dinner", icon: MapPin },
          { label: "Late night ramen spots", icon: Clock },
        ].map((chip) => (
          <button
            key={chip.label}
            onClick={() => send(chip.label)}
            className="group flex items-center gap-3 pl-4 pr-5 py-3 bg-[#151515] border border-[#2a2a2a] hover:border-orange-500/50 hover:bg-[#1a1a1a] rounded-full text-sm text-gray-300 transition-all duration-200"
          >
            <chip.icon className="w-4 h-4 text-orange-500 opacity-70 group-hover:opacity-100" />
            {chip.label}
          </button>
        ))}
      </div>
    </div>
  );

  // 2. Active Chat State (Split View)
  const renderChat = () => (
    <div className="flex flex-1 h-full overflow-hidden">
      {/* Left Column: Chat */}
      <div className={`flex-1 flex flex-col h-full overflow-hidden transition-all duration-500 ${activeSources.length > 0 ? 'lg:w-[55%] lg:flex-none' : 'w-full'}`}>
        <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-8 scrollbar-thin scrollbar-thumb-gray-800">
          {messages.map((msg, idx) => (
            <div key={idx} className={`flex gap-4 ${msg.role === "user" ? "justify-end" : "justify-start"} animate-in fade-in slide-in-from-bottom-4 duration-300`}>

              {msg.role === "assistant" && (
                <div className="w-8 h-8 rounded-full bg-[#222] border border-[#333] flex items-center justify-center flex-shrink-0 mt-1">
                  <Sparkles className="w-4 h-4 text-orange-500" />
                </div>
              )}

              <div className={`max-w-[85%] rounded-2xl p-4 leading-relaxed ${msg.role === "user"
                ? "bg-[#222] text-gray-100 rounded-tr-sm"
                : "text-gray-300 font-light"
                }`}>

                {/* --- Display Sources if available (New Feature) --- */}
                {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
                  <div className="mb-4 flex flex-wrap gap-2">
                    {msg.sources.slice(0, 3).map((s, i) => (
                      <div key={i} className="flex items-center gap-1 bg-[#1a1a1a] border border-[#333] px-2 py-1 rounded text-[10px] text-gray-500 uppercase tracking-wider">
                        <BookOpen className="w-3 h-3" />
                        <span className="truncate max-w-[100px]">{s.restaurant}</span>
                      </div>
                    ))}
                    {msg.sources.length > 3 && (
                      <div className="flex items-center px-2 py-1 text-[10px] text-gray-600">
                        +{msg.sources.length - 3} more
                      </div>
                    )}
                  </div>
                )}

                {msg.role === "assistant" && msg.content === "" ? (
                  <div className="flex items-center gap-3 text-orange-500/80">
                    <span className="relative flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-orange-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-orange-500"></span>
                    </span>
                    <span className="text-sm font-medium">Chef is cooking...</span>
                  </div>
                ) : (
                  <ReactMarkdown
                    components={{
                      strong: ({ node, ...props }) => <span className="font-semibold text-orange-400" {...props} />,
                      ul: ({ node, ...props }) => <ul className="list-disc pl-5 space-y-1 my-2" {...props} />,
                      li: ({ node, ...props }) => <li className="pl-1" {...props} />,
                      p: ({ node, ...props }) => <p className="mb-3 leading-7 last:mb-0" {...props} />,
                    }}
                  >
                    {msg.content}
                  </ReactMarkdown>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} className="h-4" />
        </div>
      </div>

      {/* Right Column: Map (Only if activeSources exist) */}
      {activeSources.length > 0 && (
        <div className="hidden lg:block flex-1 h-full border-l border-[#222] relative animate-in fade-in slide-in-from-right-4 duration-500">
          <MapPanel sources={activeSources} />
          <div className="absolute top-4 right-4 bg-black/80 backdrop-blur text-xs font-bold px-3 py-1 rounded-full border border-white/10 z-[400] text-gray-300">
            {activeSources.length} spots found
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="h-screen bg-[#0c0c0e] text-gray-100 font-sans flex flex-col overflow-hidden selection:bg-orange-500/30">

      {/* Header */}
      <header className="flex-none flex items-center gap-3 px-6 py-5 z-20 border-b border-[#1a1a1a]">
        <div className="w-10 h-10 bg-[#e86026] rounded-xl flex items-center justify-center shadow-[0_0_15px_rgba(232,96,38,0.3)]">
          <ChefHat className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-xl font-serif font-bold text-white tracking-wide">
            Local Food Guide
          </h1>
          <p className="text-[10px] text-gray-500 font-medium uppercase tracking-widest">
            Qwen 2.5 Powered Retrieval-Augmented Generation
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex overflow-hidden relative">
        {messages.length === 0 ? renderHero() : renderChat()}
      </main>

      {/* Footer Input */}
      <footer className="flex-none p-6 md:px-12 md:pb-8 bg-[#0c0c0e] z-30">
        <div className="max-w-4xl mx-auto relative group bg-[#151515] rounded-[20px] border border-[#2a2a2a] focus-within:border-[#e86026]/50 focus-within:ring-1 focus-within:ring-[#e86026]/20 transition-all duration-300 shadow-lg">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={busy}
            placeholder="Find the best spicy wings, romantic Italian, late night tacos..."
            className="w-full bg-transparent border-none text-white focus:ring-0 resize-none placeholder-gray-600 min-h-[64px] py-5 pl-6 pr-16 text-[15px] leading-relaxed"
            rows={1}
          />

          <button
            onClick={() => send()}
            disabled={busy || !input.trim()}
            className={`absolute right-3 top-3 p-2.5 rounded-xl transition-all duration-200 ${!input.trim() || busy
              ? "bg-[#222] text-gray-600"
              : "bg-[#2a2a2a] text-white hover:bg-[#e86026] hover:text-white hover:scale-105 hover:shadow-lg"
              }`}
          >
            {busy ? (
              <span className="relative flex h-5 w-5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75"></span>
              </span>
            ) : (
              <ArrowRight className="w-5 h-5" />
            )}
          </button>
        </div>

        <div className="flex justify-center items-center mt-3 px-2 text-[10px] text-gray-700 font-medium">
          <span>Press Enter to send • Powered by RAG & Qwen 2.5</span>
        </div>
      </footer>
    </div>
  );
}