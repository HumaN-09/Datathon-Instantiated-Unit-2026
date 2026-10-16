import { useState } from "react";
import { Search } from "lucide-react";
import { api, text, type AskResult } from "@/lib/api";
import { Loading, Panel } from "@/components/ui/shell";

const EXAMPLES = [
  "worst products",
  "pending orders",
  "stock",
  "category margin",
  "discount",
];

export default function Ask() {
  const [question, setQuestion] = useState("worst products");
  const [result, setResult] = useState<AskResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const run = async (q: string) => {
    setBusy(true);
    setError(null);
    try {
      setResult(await api.ask(q));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const columns = result && result.rows.length > 0 ? Object.keys(result.rows[0]) : [];

  return (
    <Loading error={error}>
      <div className="flex flex-col gap-4">
        <Panel title="Ask the database" hint="keyword intent router, SQL only">
          <form
            className="flex flex-col gap-2 sm:flex-row"
            onSubmit={(e) => {
              e.preventDefault();
              void run(question);
            }}
          >
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="worst products, pending orders, stock, category margin, discount"
              className="flex-1 rounded-[var(--radius)] border border-border bg-[rgba(20,20,20,0.75)] px-3 py-2 text-sm text-foreground outline-none placeholder:text-muted-foreground focus:border-primary/60"
            />
            <button
              type="submit"
              disabled={busy}
              className="flex items-center justify-center gap-2 rounded-[var(--radius)] border border-primary/50 bg-primary px-4 py-2 text-sm font-medium text-[#141414] disabled:opacity-60"
            >
              <Search className="h-3.5 w-3.5" />
              {busy ? "Running" : "Run"}
            </button>
          </form>
          <div className="mt-3 flex flex-wrap gap-1.5">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                type="button"
                onClick={() => {
                  setQuestion(ex);
                  void run(ex);
                }}
                className="rounded-full border border-border px-2.5 py-1 text-[11px] text-muted-foreground hover:border-primary/60 hover:text-foreground"
              >
                {ex}
              </button>
            ))}
          </div>
        </Panel>

        {result ? (
          <Panel title={`Answer (intent: ${result.question ? "matched" : "none"})`}>
            <p className="text-sm leading-relaxed text-foreground">{result.answer}</p>
          </Panel>
        ) : null}

        {result && columns.length > 0 ? (
          <Panel title="Rows retrieved">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="text-muted-foreground">
                  <tr>
                    {columns.map((c) => (
                      <th key={c} className="py-2 pr-3 font-medium">
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.rows.map((row, i) => (
                    <tr key={i} className="border-t border-border">
                      {columns.map((c) => (
                        <td key={c} className="py-2 pr-3 tabular-nums text-foreground">
                          {text(row[c])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        ) : null}
      </div>
    </Loading>
  );
}
