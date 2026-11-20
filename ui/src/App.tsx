import { useEffect, useState } from "react";
import Dashboard from "./components/Dashboard";
import Compare from "./components/Compare";
import Overview, { Go } from "./components/Overview";
import SourceSelector from "./components/SourceSelector";
import SystemStatus from "./components/SystemStatus";
import { Activity } from "lucide-react";

type Tab = "overview" | "live" | "compare";
const TABS: { key: Tab; label: string }[] = [
  { key: "overview", label: "Overview" },
  { key: "live", label: "Live monitor" },
  { key: "compare", label: "CPU · T4 · Jetson" },
];
// Each view is linkable: #live, #compare; no hash opens the overview.
const fromHash = (): Tab => {
  const h = location.hash.slice(1);
  return h === "compare" || h === "live" ? h : "overview";
};

export default function App() {
  const [tab, setTab] = useState<Tab>(fromHash);
  useEffect(() => {
    const on = () => setTab(fromHash());
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);

  const go: Go = (t, then) => {
    history.replaceState(null, "", t === "overview" ? location.pathname : `#${t}`);
    setTab(t);
    // Jump to a panel of the live monitor once it is shown.
    const target = then === "log" ? "activity-log" : then === "people" ? "known-people" : null;
    requestAnimationFrame(() => {
      if (target) document.getElementById(target)?.scrollIntoView({ behavior: "smooth", block: "start" });
      else scrollTo({ top: 0 });
    });
  };
  const open = (t: Tab) => go(t);

  const tabClass = (t: Tab) =>
    "px-3 py-1.5 rounded-md text-sm font-medium transition " +
    (tab === t ? "bg-accent/15 text-accent" : "text-muted hover:text-slate-100");

  return (
    <div className="min-h-full flex flex-col">
      <header className="sticky top-0 z-40 border-b border-border bg-elevated/95 backdrop-blur">
        <div className="max-w-[1600px] mx-auto px-6 py-3 flex flex-wrap items-center gap-x-6 gap-y-3">
          <button onClick={() => open("overview")} className="flex items-center gap-3 text-left">
            <div className="w-9 h-9 rounded-lg bg-accent/20 border border-accent/40 flex items-center justify-center">
              <Activity size={18} className="text-accent" />
            </div>
            <div>
              <h1 className="text-lg font-semibold leading-tight">Human Action & Behavior Monitor</h1>
              <p className="text-xs text-muted">Activity recognition · face recognition · proof images</p>
            </div>
          </button>
          <nav className="flex gap-1" aria-label="Views">
            {TABS.map((t) => (
              <button key={t.key} className={tabClass(t.key)} onClick={() => open(t.key)}
                aria-current={tab === t.key ? "page" : undefined}>
                {t.label}
              </button>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-6">
            {tab === "live" && <SourceSelector />}
            <SystemStatus />
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-[1600px] w-full mx-auto p-6">
        {/* The live view stays mounted so its video and socket are not torn
            down by a look at the other views. */}
        <div hidden={tab !== "live"}><Dashboard /></div>
        {tab === "overview" && <Overview go={go} />}
        {tab === "compare" && <Compare />}
      </main>
    </div>
  );
}
