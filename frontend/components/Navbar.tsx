"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, Brain, BarChart3, Info, ShieldAlert, Sparkles, Menu, X } from "lucide-react";
import { checkHealth } from "@/lib/api";

export default function Navbar() {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [engineOnline, setEngineOnline] = useState<boolean | null>(null);

  useEffect(() => {
    checkHealth()
      .then((res) => setEngineOnline(res.status === "healthy"))
      .catch(() => setEngineOnline(false));

    const interval = setInterval(() => {
      checkHealth()
        .then((res) => setEngineOnline(res.status === "healthy"))
        .catch(() => setEngineOnline(false));
    }, 15000);

    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { name: "Classifier", href: "/classify", icon: Brain },
    { name: "Analytics", href: "/analytics", icon: BarChart3 },
    { name: "Model Architecture", href: "/models", icon: Activity },
    { name: "Responsible AI", href: "/about", icon: ShieldAlert },
  ];

  return (
    <header className="sticky top-0 z-50 w-full glass-panel border-b border-slate-800/80 bg-slate-950/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Logo & Brand */}
          <Link href="/" className="flex items-center gap-3 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-600 to-accent-500 p-0.5 flex items-center justify-center shadow-lg shadow-brand-500/20 group-hover:shadow-brand-500/40 transition-all">
              <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
                <Brain className="w-5 h-5 text-brand-400 group-hover:scale-110 transition-transform" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg text-white tracking-tight">MedClass</span>
                <span className="text-xs font-semibold px-1.5 py-0.5 rounded bg-brand-500/20 text-brand-300 border border-brand-500/30">AI</span>
              </div>
              <p className="text-[10px] text-slate-400 hidden sm:block tracking-wide">Question Intelligence</p>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-all ${
                    isActive
                      ? "text-brand-300 bg-brand-950/60 border border-brand-500/30 shadow-sm shadow-brand-500/10"
                      : "text-slate-300 hover:text-white hover:bg-slate-800/50"
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? "text-brand-400" : "text-slate-400"}`} />
                  {item.name}
                </Link>
              );
            })}
          </nav>

          {/* Right Action & Engine Status */}
          <div className="hidden sm:flex items-center gap-4">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-xs text-slate-300">
              <span className={`w-2 h-2 rounded-full ${
                engineOnline === true
                  ? "bg-emerald-400 animate-pulse shadow-[0_0_8px_#34d399]"
                  : engineOnline === false
                  ? "bg-rose-400"
                  : "bg-amber-400"
              }`} />
              <span>{engineOnline ? "Transformer Online" : "Checking Engine..."}</span>
            </div>

            <Link
              href="/classify"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-brand-600 to-accent-600 hover:from-brand-500 hover:to-accent-500 text-white text-xs font-semibold tracking-wide shadow-md shadow-brand-500/20 hover:shadow-brand-500/40 transition-all active:scale-95"
            >
              <Sparkles className="w-3.5 h-3.5" />
              Try Classifier
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <div className="flex items-center gap-2 md:hidden">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/60 focus:outline-none"
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile dropdown */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t border-slate-800 bg-slate-950/95 px-4 pt-3 pb-5 space-y-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setMobileMenuOpen(false)}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-base font-medium ${
                  isActive
                    ? "text-brand-300 bg-brand-950/70 border border-brand-500/30"
                    : "text-slate-300 hover:text-white hover:bg-slate-800/60"
                }`}
              >
                <Icon className="w-5 h-5 text-brand-400" />
                {item.name}
              </Link>
            );
          })}
          <div className="pt-3 border-t border-slate-800 flex flex-col gap-2">
            <div className="flex items-center gap-2 px-3 py-1.5 text-xs text-slate-400">
              <span className={`w-2 h-2 rounded-full ${engineOnline ? "bg-emerald-400" : "bg-amber-400"}`} />
              <span>{engineOnline ? "Engine Online • v2.0" : "Connecting..."}</span>
            </div>
            <Link
              href="/classify"
              onClick={() => setMobileMenuOpen(false)}
              className="flex items-center justify-center gap-2 w-full py-2.5 rounded-lg bg-brand-600 text-white font-medium text-sm shadow-md"
            >
              <Sparkles className="w-4 h-4" />
              Try Classifier Now
            </Link>
          </div>
        </div>
      )}
    </header>
  );
}
