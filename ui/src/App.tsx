import { useEffect, useState } from "react";
import Dashboard from "./components/Dashboard";
import Compare from "./components/Compare";
import SourceSelector from "./components/SourceSelector";
import SystemStatus from "./components/SystemStatus";
import { Activity } from "lucide-react";

type Tab = "live" | "compare";
const fromHash = (): Tab => (location.hash === "#compare" ? "compare" : "live");

export default function App() {
  // Two views, each linkable: #compare opens the CPU-vs-GPU comparison.
  const [tab, setTab] = useState<Tab>(fromHash);
  useEffect(() => {
    const on = () => setTab(fromHash());
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  const go = (t: Tab) => { location.hash = t === "compare" ? "compare" : ""; setTab(t); };

  const tabClass = (t: Tab) =>
    "px-3 py-1.5 rounded-md text-sm font-medium transition " +
    (tab === t ? "bg-accent/15 text-accent" : "text-muted hover:text-slate-100");

  return (
    <div className="min-h-full flex flex-col">
      <header className="border-b border-border bg-elevated">
        <div className="max-w-[1600px] mx-auto px-6 py-4 flex flex-wrap items-center gap-x-6 gap-y-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-accent/20 border border-accent/40 flex items-center justify-center">
              <Activity size={18} className="text-accent" />
            </div>
            <div>
              <h1 className="text-lg font-semibold">Human Action & Behavior Monitor</h1>
              <p className="text-xs text-muted">Real-time detection · tracking · action recognition</p>
            </div>
          </div>
          <nav className="flex gap-1" aria-label="Views">
            <button className={tabClass("live")} onClick={() => go("live")}>Live monitor</button>
            <button className={tabClass("compare")} onClick={() => go("compare")}>CPU vs GPU</button>
          </nav>
          <div className="ml-auto flex items-center gap-6">
            {tab === "live" && <SourceSelector />}
            <SystemStatus />
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-[1600px] w-full mx-auto p-6">
        {/* The live view stays mounted so its video and socket are not torn
            down by a look at the comparison. */}
        <div hidden={tab !== "live"}><Dashboard /></div>
        {tab === "compare" && <Compare />}
      </main>
    </div>
  );
}
