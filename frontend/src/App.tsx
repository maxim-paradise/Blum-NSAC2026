import { Moon, PanelLeftOpen, SlidersHorizontal, Sparkles, Sun } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { CatalogSection } from "./components/dashboard/CatalogSection";
import { CompareSection } from "./components/dashboard/CompareSection";
import { GapMapSection } from "./components/dashboard/GapMapSection";
import { HeaderKpis } from "./components/dashboard/HeaderKpis";
import { OxygenChart } from "./components/dashboard/OxygenChart";
import { RankingSection } from "./components/dashboard/RankingSection";
import { SeriesCardSection } from "./components/dashboard/SeriesCardSection";
import { Sidebar } from "./components/sidebar/Sidebar";
import { useAgentChat } from "./hooks/useAgentChat";
import { useDashboard } from "./hooks/useDashboard";
import { useTheme } from "./hooks/useTheme";

export function App() {
  const { data, loading, error, filters, setFilters, resetFilters, maxTests, filteredSeries } =
    useDashboard();
  const agent = useAgentChat();
  const { isDark, toggleTheme } = useTheme();

  const [activeTab, setActiveTab] = useState<"filters" | "ai">("filters");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() =>
    typeof window !== "undefined" ? window.innerWidth < 768 : false,
  );
  const [selectedCardId, setSelectedCardId] = useState<number>(1);
  const seriesCardRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mq = window.matchMedia("(max-width: 767px)");
    const onChange = (e: MediaQueryListEvent) => {
      setSidebarCollapsed(e.matches);
    };
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  const filteredIds = useMemo(() => new Set(filteredSeries.map((s) => s.id)), [filteredSeries]);

  const totalTests = useMemo(
    () => (data ? data.series.reduce((acc, s) => acc + s.row_count, 0) : 0),
    [data],
  );
  const moonSeriesTotal = useMemo(
    () => (data ? data.series.filter((s) => s.o2.max !== null && s.o2.max >= 30).length : 0),
    [data],
  );

  const handleCatalogSelect = (id: number) => {
    setSelectedCardId(id);
    seriesCardRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-[var(--bg-page)] text-sm text-[var(--ink-muted)]">
        Loading NASA PSI dataset...
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex h-screen items-center justify-center bg-[var(--bg-page)] p-6">
        <div className="max-w-md rounded-2xl border border-[var(--danger)]/25 bg-[var(--danger-soft)] p-5 text-sm text-[var(--danger)]">
          {error || "Could not load dataset."}
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-dvh w-screen overflow-hidden bg-[var(--bg-page)] text-[var(--ink)]">
      <Sidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        collapsed={sidebarCollapsed}
        onCollapse={() => setSidebarCollapsed(true)}
        isDark={isDark}
        onToggleTheme={toggleTheme}
        filteredCount={filteredSeries.length}
        totalCount={data.series.length}
        filters={filters}
        maxTests={maxTests}
        onFiltersChange={setFilters}
        onResetFilters={resetFilters}
        agent={agent}
      />

      <main className="psi-scroll relative flex-1 min-w-0 h-dvh overflow-y-auto overflow-x-hidden">
        {sidebarCollapsed && (
          <div className="sticky top-0 z-20 flex items-center justify-between gap-2 border-b border-[var(--border)] bg-[var(--bg-page)]/90 px-4 py-2.5 backdrop-blur md:fixed md:top-4 md:left-4 md:border-none md:bg-transparent md:p-0 md:backdrop-blur-none">
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => {
                  setActiveTab("filters");
                  setSidebarCollapsed(false);
                }}
                aria-label="Open sidebar"
                className="psi-btn inline-flex items-center gap-1.5 rounded-lg border border-[var(--border-strong)] bg-[var(--bg-card)]/95 px-2.5 py-1.5 text-xs font-medium text-[var(--ink)] backdrop-blur hover:bg-[var(--bg-elevated)]"
              >
                <PanelLeftOpen className="hidden h-3.5 w-3.5 text-[var(--action)] md:inline" />
                <SlidersHorizontal className="h-3.5 w-3.5 text-[var(--action)] md:hidden" />
                <span className="hidden md:inline">Explore PSI</span>
                <span className="md:hidden">Filters</span>
                <span className="rounded bg-[var(--bg-subtle)] px-1 py-0.2 font-mono text-[10.5px] text-[var(--ink-muted)] tabular-nums md:hidden">
                  {filteredSeries.length}/{data.series.length}
                </span>
              </button>
              {agent.configured && (
                <button
                  type="button"
                  onClick={() => {
                    setActiveTab("ai");
                    setSidebarCollapsed(false);
                  }}
                  aria-label="Open AI Agent"
                  className="psi-btn inline-flex items-center gap-1.5 rounded-lg border border-[var(--border-strong)] bg-[var(--bg-card)]/95 px-2.5 py-1.5 text-xs font-medium text-[var(--ink)] hover:bg-[var(--bg-elevated)] md:hidden"
                >
                  <Sparkles className="h-3.5 w-3.5 text-[var(--action)]" />
                  AI Agent
                </button>
              )}
            </div>
            <button
              type="button"
              onClick={toggleTheme}
              aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
              className="psi-btn rounded-lg border border-[var(--border-strong)] bg-[var(--bg-card)]/95 p-1.5 text-[var(--ink-muted)] backdrop-blur hover:bg-[var(--bg-elevated)] hover:text-[var(--ink)]"
            >
              {isDark ? <Sun className="h-3.5 w-3.5 text-amber-400" /> : <Moon className="h-3.5 w-3.5" />}
            </button>
          </div>
        )}

        <div className="mx-auto max-w-[1200px] space-y-5 px-4 py-5 sm:space-y-6 sm:px-8 sm:py-8">
          <HeaderKpis
            totalSeries={data.series.length}
            totalTests={totalTests}
            moonSeriesTotal={moonSeriesTotal}
            filtered={filteredSeries}
          />

          {filteredSeries.length === 0 ? (
            <div className="psi-card p-6 text-sm text-[var(--ink-secondary)]">
              No series match these filters. Reset filters in the sidebar to view all 22 series.
            </div>
          ) : (
            <>
              <OxygenChart series={filteredSeries} isDark={isDark} />
              <GapMapSection
                series={filteredSeries}
                paramOrder={data.param_order}
                filters={filters}
              />
              <RankingSection
                goals={data.goals}
                rankings={data.rankings}
                filteredIds={filteredIds}
              />
              <CompareSection series={filteredSeries} paramOrder={data.param_order} />
              <CatalogSection series={filteredSeries} onSelectSeries={handleCatalogSelect} />
              <div ref={seriesCardRef}>
                <SeriesCardSection
                  series={filteredSeries}
                  selectedId={selectedCardId}
                  onSelectId={setSelectedCardId}
                />
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
