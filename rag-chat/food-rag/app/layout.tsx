import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Local Food Guide",
  description: "AI-powered restaurant discovery",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body
        className={`
          ${geistSans.variable} 
          ${geistMono.variable}
          antialiased
          min-h-screen
          bg-gradient-to-br from-gray-950 via-gray-900 to-black
          text-gray-100
        `}
      >
        {/* Global container */}
        <div className="relative flex min-h-screen flex-col">
          {/* Subtle background glow */}
          <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(ellipse_at_top,_rgba(249,115,22,0.15),_transparent_60%)]" />

          {/* App content */}
          <main className="relative z-10 flex-1">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
