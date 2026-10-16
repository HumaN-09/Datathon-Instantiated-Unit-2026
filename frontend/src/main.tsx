import React from "react";
import ReactDOM from "react-dom/client";
import App from "@/App";
import { KineticGrid } from "@/components/ui/kinetic-grid";
import "@/index.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <KineticGrid>
      <App />
    </KineticGrid>
  </React.StrictMode>,
);
