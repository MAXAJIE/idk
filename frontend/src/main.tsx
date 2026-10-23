import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import Spectator from "./pages/Spectator";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Spectator />
  </StrictMode>,
);
