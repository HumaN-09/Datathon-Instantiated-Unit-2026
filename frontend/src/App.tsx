import { useState } from "react";
import { BarChart3, MessageSquare, Sparkles, Table2 } from "lucide-react";
import { cn } from "@/lib/utils";
import Dashboard from "@/components/Dashboard";
import Predict from "@/components/Predict";
import Ask from "@/components/Ask";
import Evidence from "@/components/Evidence";

const TABS = [
  { id: "dashboard", label: "Dashboard", icon: BarChart3 },
  { id: "predict", label: "Predict", icon: Sparkles },
  { id: "ask", label: "Ask", icon: MessageSquare },
  { id: "evidence", label: "Model Evidence", icon: Table2 },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function App() {
  const [tab, setTab] = useState<TabId>("dashboard");

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-[1180px] flex-col px-4 pb-6 pt-5 sm:px-6">
      <header className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-foreground">Money Leak</h1>
          <p className="text-xs text-muted-foreground">
            Where money leaks, and what to sell, at what price, with what discount.
          </p>
        </div>
        <nav className="flex flex-wrap gap-1 rounded-full border border-border bg-[rgba(28,28,28,0.7)] p-1 backdrop-blur">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              type="button"
              onClick={() => setTab(id)}
              className={cn(
                "flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium transition-colors",
                tab === id
                  ? "bg-primary text-[#141414]"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              {label}
            </button>
          ))}
        </nav>
      </header>

      <main className="flex-1">
        {tab === "dashboard" ? <Dashboard /> : null}
        {tab === "predict" ? <Predict /> : null}
        {tab === "ask" ? <Ask /> : null}
        {tab === "evidence" ? <Evidence /> : null}
      </main>

      <footer className="mt-8 text-center text-[11px] text-muted-foreground">
        Team Instantiated Unit by S. Ariyan Minaxi and Tanush Kumar
      </footer>
    </div>
  );
}
