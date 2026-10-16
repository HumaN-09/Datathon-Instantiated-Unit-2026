import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

/**
 * LayoutGrid: an expanding card grid. Clicking a card lifts it out of the grid
 * into a viewport-centred panel with a shared-element transition
 * (framer-motion layoutId); the backdrop, or Escape, puts it back.
 *
 * Adapted from the 21st.dev component for this app:
 *   - no App Router directives and no next/image; a plain lazy <img> instead,
 *     because this is a Vite + React SPA, not Next.js;
 *   - `content` is React.ReactNode rather than JSX.Element, and the unused
 *     demo imports were dropped because this project builds with noUnusedLocals;
 *   - `fit`, `label` and `badge` were added so saved matplotlib charts (which
 *     must never be cropped) can be shown contained, captioned and labelled;
 *   - `className` lets the grid lose the demo padding when nested in a panel;
 *   - the expanded card is rendered through a portal with `position: fixed`, so
 *     it is centred on the viewport. The original sized it from its container,
 *     which puts a chart clicked in the tenth row off-screen, and any ancestor
 *     with backdrop-filter (every Panel here) would otherwise become the
 *     containing block for a fixed child;
 *   - the caption is a footer bar under the chart instead of a scrim over it,
 *     so long chart captions never hide the axes.
 */
export type GridCard = {
  /** Must be unique across the whole page: it keys the shared-element animation. */
  id: number;
  content: React.ReactNode;
  className: string;
  thumbnail: string;
  alt?: string;
  /** "cover" crops (photos), "contain" letterboxes (charts). Default "cover". */
  fit?: "cover" | "contain";
  /** Always-visible caption on the thumbnail, hidden while expanded. */
  label?: string;
  /** Corner affordance, e.g. a lucide Maximize2 icon. */
  badge?: React.ReactNode;
};

export function LayoutGrid({
  cards,
  className,
}: {
  cards: GridCard[];
  className?: string;
}) {
  const [selected, setSelected] = useState<GridCard | null>(null);

  const handleClick = (card: GridCard) => setSelected(card);
  const handleOutsideClick = () => setSelected(null);

  useEffect(() => {
    if (!selected) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") handleOutsideClick();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected]);

  return (
    <>
      <div
        className={cn(
          "relative mx-auto grid h-full w-full max-w-7xl grid-cols-1 gap-4 p-10 md:grid-cols-3",
          className,
        )}
      >
        {cards.map((card) => {
          const isSelected = selected?.id === card.id;
          return (
            <div key={card.id} className={cn(card.className, "")}>
              {isSelected ? (
                // The grid slot stays occupied, and the same layoutId is free
                // for the panel, so the card animates out of its own cell.
                <div className="h-full w-full rounded-xl bg-white/20" />
              ) : (
                <motion.div
                  layoutId={`card-${card.id}`}
                  onClick={() => handleClick(card)}
                  className={cn(card.className, "group relative cursor-pointer overflow-hidden rounded-xl bg-white")}
                >
                  <ImageComponent card={card} />
                  {card.badge ? (
                    <span className="absolute right-2 top-2 z-20 rounded-md bg-black/60 p-1 text-white opacity-0 transition-opacity group-hover:opacity-100">
                      {card.badge}
                    </span>
                  ) : null}
                  {card.label ? (
                    <span className="absolute bottom-2 left-2 z-20 max-w-[85%] truncate rounded-full bg-black/70 px-2 py-0.5 text-[11px] font-medium text-white">
                      {card.label}
                    </span>
                  ) : null}
                </motion.div>
              )}
            </div>
          );
        })}
        <motion.div
          onClick={handleOutsideClick}
          className={cn(
            "absolute left-0 top-0 z-10 h-full w-full bg-black opacity-0",
            selected?.id ? "pointer-events-auto" : "pointer-events-none",
          )}
          animate={{ opacity: selected?.id ? 0.3 : 0 }}
        />
      </div>

      {typeof document !== "undefined"
        ? createPortal(
            <>
              {selected ? (
                <motion.div
                  onClick={handleOutsideClick}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 0.72 }}
                  className="fixed inset-0 z-[80] bg-black"
                />
              ) : null}
              {selected ? (
                <div className="pointer-events-none fixed inset-0 z-[90] flex items-center justify-center p-3 md:p-6">
                  <motion.div
                    layoutId={`card-${selected.id}`}
                    role="dialog"
                    aria-modal="true"
                    aria-label={selected.alt ?? "chart"}
                    className="pointer-events-auto relative flex h-[78vh] w-full max-w-6xl flex-col overflow-hidden rounded-xl bg-white md:h-[72vh] md:w-[76vw]"
                  >
                    {/* The chart gets the whole area above the caption bar, so
                        the caption can never cover the x-axis labels. */}
                    <div className="relative flex-1 overflow-hidden">
                      <ImageComponent card={selected} />
                    </div>
                    <SelectedCard selected={selected} />
                  </motion.div>
                </div>
              ) : null}
            </>,
            document.body,
          )
        : null}
    </>
  );
}

const ImageComponent = ({ card, className }: { card: GridCard; className?: string }) => {
  return (
    <motion.img
      layoutId={`image-${card.id}-image`}
      src={card.thumbnail}
      alt={card.alt ?? card.label ?? "chart"}
      loading="lazy"
      className={cn(
        "absolute inset-0 h-full w-full transition duration-200",
        card.fit === "contain" ? "object-contain" : "object-cover object-top",
        className,
      )}
    />
  );
};

const SelectedCard = ({ selected }: { selected: GridCard | null }) => {
  return (
    <div className="relative z-[60] shrink-0 border-t border-black/10 bg-[#101010] px-6 py-3 md:px-8">
      <motion.div
        layoutId={`content-${selected?.id}`}
        initial={{ opacity: 0, y: 100 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: "easeInOut" }}
        className="relative z-[70]"
      >
        {selected?.content}
      </motion.div>
    </div>
  );
};

export default LayoutGrid;
