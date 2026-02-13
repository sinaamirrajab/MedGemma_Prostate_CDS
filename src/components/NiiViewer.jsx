import { useEffect, useRef, useState } from "react";
import { Niivue } from "@niivue/niivue";

export default function NiiViewer({ title, subtitle, url }) {
  const canvasRef = useRef(null);
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    const nv = new Niivue();

    const init = async () => {
      try {
        await nv.attachToCanvas(canvasRef.current);
        await nv.loadVolumes([
          {
            url,
            colorMap: "gray",
          },
        ]);

        if (!active) return;

        if (typeof nv.setSliceType === "function") {
          nv.setSliceType(nv.sliceTypeAxial);
        }

        setStatus("ready");
      } catch (loadError) {
        if (!active) return;
        setError(loadError?.message ?? "Failed to load volume.");
        setStatus("error");
      }
    };

    init();

    return () => {
      active = false;
      if (typeof nv.destroy === "function") {
        nv.destroy();
      }
    };
  }, [url]);

  return (
    <div className="image-panel">
      <div className="image-placeholder image-canvas niivue">
        <canvas ref={canvasRef} aria-label={`${title} volume`} />
        {status !== "ready" ? (
          <div className="viewer-overlay">
            <span>{title}</span>
            <p>{status === "error" ? "Unable to load" : "Loading volume..."}</p>
          </div>
        ) : (
          <div className="viewer-hint">Scroll or drag to navigate slices</div>
        )}
      </div>
      <div className="image-meta">
        <span>{subtitle}</span>
        <span>{title}</span>
      </div>
      {error ? <div className="viewer-error">{error}</div> : null}
    </div>
  );
}
