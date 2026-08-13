import { useState, useRef, useEffect } from "react";

/* ------------------------------------------------------------------
   BACKEND CONTRACT

   POST /api/chat
   request  { session_id: string, message: string }
   response {
     session_id: string,
     reply: string,                       // what the agent says to the user
     trace: [                             // ReAct steps, in order
       { thought: string,
         tool: string | null,             // "search_restaurants" | "check_availability" | "make_reservation"
         tool_input: object | null,
         observation: string | null }
     ],
     restaurants?: [ { id, name, cuisine, location, avg_price, rating, hours } ],
     done?: boolean                       // true once a reservation is confirmed
   }

   Set USE_MOCK = false once the endpoint exists.
------------------------------------------------------------------- */

const USE_MOCK = true;
const API_URL = "/api/chat";

const EXAMPLES = [
  "Table for four in Nanshan tomorrow at 7pm, Korean BBQ.",
  "Cheap Singaporean food near Futian, two people, Friday lunch.",
  "Somewhere quiet for a business dinner under ¥100 a head.",
];

const TOOL_LABELS = {
  search_restaurants: "Searching restaurants",
  check_availability: "Checking availability",
  make_reservation: "Making reservation",
};

/* ---------------------------- mock agent ---------------------------- */

const MOCK_TURNS = [
  {
    trace: [
      {
        thought:
          "The user wants Korean food for four in Nanshan tomorrow evening. I need candidate restaurants before I can check any tables.",
        tool: "search_restaurants",
        tool_input: { cuisine: "korean", location: "Nanshan", party_size: 4 },
        observation: "2 restaurants matched: Hanok Table (R008), Bunsik Alley (R007).",
      },
      {
        thought:
          "Hanok Table has the higher rating and takes parties of four. I will check the requested 19:00 slot first.",
        tool: "check_availability",
        tool_input: { restaurant_id: "R008", date: "2026-08-15", time: "19:00", party_size: 4 },
        observation: "0 tables available at 19:00.",
      },
      {
        thought:
          "19:00 is full. Rather than stopping, I will scan the slots on either side to give the user a real choice.",
        tool: "check_availability",
        tool_input: { restaurant_id: "R008", date: "2026-08-15", time: ["18:30", "19:30", "20:00"] },
        observation: "18:30 → 4 tables, 19:30 → 3 tables, 20:00 → 6 tables.",
      },
    ],
    reply:
      "Hanok Table in Nanshan is booked solid at 19:00 tomorrow. There is room at 18:30, 19:30 and 20:00 for four people. Which one works?",
    restaurants: [
      {
        id: "R008",
        name: "Hanok Table",
        cuisine: "Korean",
        location: "Nanshan",
        avg_price: 52,
        rating: 4.5,
        hours: "17:00 – 21:30",
      },
    ],
  },
  {
    trace: [
      {
        thought:
          "The user picked 19:30, which the last observation confirmed has three tables free. I can book it directly.",
        tool: "make_reservation",
        tool_input: {
          restaurant_id: "R008",
          date: "2026-08-15",
          time: "19:30",
          party_size: 4,
        },
        observation: "Reservation RSV0007 created. Tables remaining at 19:30: 2.",
      },
    ],
    reply:
      "Booked. Hanok Table, tomorrow at 19:30, table for four, under confirmation RSV0007. They hold the table for fifteen minutes.",
    done: true,
  },
];

function mockChat(turnIndex) {
  const turn = MOCK_TURNS[Math.min(turnIndex, MOCK_TURNS.length - 1)];
  return new Promise((resolve) => setTimeout(() => resolve(turn), 1400));
}

/* ------------------------------ trace ------------------------------ */

function Trace({ steps }) {
  const [open, setOpen] = useState(false);
  if (!steps || steps.length === 0) return null;

  return (
    <div className="trace">
      <button className="trace-toggle" type="button" onClick={() => setOpen(!open)}>
        <span className="trace-count">{steps.length}</span>
        {open ? "Hide reasoning" : "Show reasoning"}
      </button>

      {open && (
        <ol className="docket">
          {steps.map((s, i) => (
            <li key={i} className="docket-step">
              <div className="docket-num">{String(i + 1).padStart(2, "0")}</div>
              <div className="docket-body">
                <p className="docket-thought">{s.thought}</p>
                {s.tool && (
                  <div className="docket-call">
                    <span className="docket-tool">{s.tool}</span>
                    <code>{JSON.stringify(s.tool_input)}</code>
                  </div>
                )}
                {s.observation && <p className="docket-obs">{s.observation}</p>}
              </div>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

/* ---------------------------- mini cards ---------------------------- */

function ResultStrip({ restaurants }) {
  if (!restaurants || restaurants.length === 0) return null;
  return (
    <ul className="strip">
      {restaurants.map((r) => (
        <li key={r.id} className="mini">
          <div className="mini-top">
            <span className="mini-name">{r.name}</span>
            <span className="mini-score">{r.rating.toFixed(1)}</span>
          </div>
          <p className="mini-meta">
            {r.cuisine} · {r.location} · ¥{r.avg_price} pp
          </p>
          <p className="mini-hours">{r.hours}</p>
        </li>
      ))}
    </ul>
  );
}

/* ------------------------------ app ------------------------------ */

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(null); // current tool label
  const [error, setError] = useState(null);
  const [turn, setTurn] = useState(0);
  const sessionId = useRef(
    `s_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`
  );
  const threadRef = useRef(null);
  const taRef = useRef(null);

  const started = messages.length > 0;

  useEffect(() => {
    if (threadRef.current) {
      threadRef.current.scrollTop = threadRef.current.scrollHeight;
    }
  }, [messages, pending]);

  const grow = () => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 180) + "px";
  };

  const send = async (text) => {
    const body = text.trim();
    if (!body || pending) return;

    setMessages((m) => [...m, { role: "user", text: body }]);
    setInput("");
    setError(null);
    setPending("Reading your request");
    if (taRef.current) taRef.current.style.height = "auto";

    try {
      let data;
      if (USE_MOCK) {
        setTimeout(() => setPending(TOOL_LABELS.search_restaurants), 500);
        data = await mockChat(turn);
        setTurn((t) => t + 1);
      } else {
        const res = await fetch(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId.current, message: body }),
        });
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        data = await res.json();
      }

      setMessages((m) => [
        ...m,
        {
          role: "agent",
          text: data.reply,
          trace: data.trace,
          restaurants: data.restaurants,
          done: data.done,
        },
      ]);
    } catch (e) {
      setError(
        e.message === "Failed to fetch"
          ? "Can't reach the agent. Check that the backend is running on port 8000."
          : e.message
      );
    } finally {
      setPending(null);
    }
  };

  const reset = () => {
    setMessages([]);
    setTurn(0);
    setError(null);
    setInput("");
    sessionId.current = `s_${Date.now().toString(36)}`;
  };

  return (
    <div className="app">
      <style>{`
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Instrument+Sans:wght@400;500;600&family=Space+Mono:wght@400;700&display=swap');

.app{
  --ink:#14232E;
  --ink-soft:#5A6A73;
  --paper:#E4EAE4;
  --card:#FBFAF6;
  --jade:#17594A;
  --chili:#C8402C;
  --amber:#E4A93C;
  --line:rgba(20,35,46,.16);
  --display:'Bricolage Grotesque',system-ui,sans-serif;
  --body:'Instrument Sans',system-ui,sans-serif;
  --mono:'Space Mono',ui-monospace,monospace;
  background:var(--paper);color:var(--ink);font-family:var(--body);
  min-height:100vh;display:flex;flex-direction:column;
}
.app *{box-sizing:border-box}
.app button{font:inherit;color:inherit;cursor:pointer}
.app :focus-visible{outline:3px solid var(--jade);outline-offset:3px}

/* header */
.bar{
  display:flex;align-items:center;justify-content:space-between;gap:16px;
  padding:16px 24px;border-bottom:1px solid var(--line);
  position:sticky;top:0;background:var(--paper);z-index:4;
}
.brand{display:flex;align-items:baseline;gap:12px;min-width:0}
.brand-mark{font-family:var(--mono);font-size:11px;letter-spacing:.2em;text-transform:uppercase;background:var(--jade);color:var(--card);padding:5px 9px}
.brand-name{font-family:var(--display);font-weight:700;font-size:17px;letter-spacing:-.015em;white-space:nowrap}
.reset{background:none;border:1px solid var(--line);padding:7px 12px;font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-soft)}
.reset:hover{border-color:var(--ink);color:var(--ink)}

/* thread */
.thread{flex:1;overflow-y:auto;padding:0 24px}
.thread-inner{max-width:760px;margin:0 auto;padding:36px 0 24px;display:flex;flex-direction:column;gap:22px}

/* opening */
.opening{padding:52px 0 8px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--jade);margin:0 0 16px}
.title{font-family:var(--display);font-weight:800;font-size:clamp(34px,6vw,60px);line-height:.96;letter-spacing:-.03em;margin:0}
.lede{max-width:46ch;margin:18px 0 0;color:var(--ink-soft);font-size:16px;line-height:1.55}
.examples{margin:30px 0 0;display:flex;flex-direction:column;gap:9px}
.ex-label{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink-soft);margin:0 0 3px}
.ex{
  text-align:left;background:var(--card);border:1px solid var(--line);padding:14px 16px;
  font-size:15px;line-height:1.4;display:flex;gap:12px;align-items:baseline;
  transition:transform .12s ease,box-shadow .12s ease,border-color .12s ease;
}
.ex:hover{transform:translateY(-2px);box-shadow:0 6px 0 -2px var(--line);border-color:var(--jade)}
.ex-arrow{font-family:var(--mono);color:var(--jade);flex:none}

/* messages */
.msg{display:flex;flex-direction:column;gap:8px}
.who{font-family:var(--mono);font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:var(--ink-soft)}
.bubble{padding:16px 18px;font-size:15px;line-height:1.55}
.msg-user{align-items:flex-end}
.msg-user .bubble{background:var(--ink);color:var(--card);max-width:80%}
.msg-agent .bubble{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--jade)}
.msg-agent.is-done .bubble{border-left-color:var(--amber)}
.confirm{
  display:inline-block;margin-top:12px;font-family:var(--mono);font-size:11px;
  letter-spacing:.14em;text-transform:uppercase;background:var(--amber);color:var(--ink);padding:5px 9px;
}

/* docket / trace */
.trace{margin-top:2px}
.trace-toggle{
  background:none;border:0;padding:0;display:inline-flex;align-items:center;gap:8px;
  font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--jade);text-decoration:underline;text-underline-offset:4px;
}
.trace-count{background:var(--jade);color:var(--card);padding:2px 6px;text-decoration:none;letter-spacing:0}
.docket{
  list-style:none;margin:12px 0 0;padding:16px;background:var(--card);
  border:1px dashed var(--line);display:flex;flex-direction:column;gap:16px;
}
.docket-step{display:flex;gap:14px}
.docket-num{font-family:var(--mono);font-weight:700;font-size:12px;color:var(--chili);flex:none;padding-top:2px}
.docket-body{min-width:0;display:flex;flex-direction:column;gap:8px}
.docket-thought{margin:0;font-size:14px;line-height:1.5}
.docket-call{display:flex;flex-wrap:wrap;align-items:center;gap:8px}
.docket-tool{font-family:var(--mono);font-size:11px;background:var(--jade);color:var(--card);padding:4px 8px;letter-spacing:.04em}
.docket-call code{font-family:var(--mono);font-size:12px;color:var(--ink-soft);word-break:break-all}
.docket-obs{
  margin:0;font-family:var(--mono);font-size:12px;line-height:1.5;
  padding:8px 10px;background:var(--paper);border-left:2px solid var(--chili);
}

/* result strip */
.strip{list-style:none;margin:12px 0 0;padding:0;display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px}
.mini{background:var(--card);border:1px solid var(--line);padding:14px}
.mini-top{display:flex;align-items:baseline;justify-content:space-between;gap:10px}
.mini-name{font-family:var(--display);font-weight:700;font-size:16px;letter-spacing:-.01em}
.mini-score{font-family:var(--mono);font-weight:700;font-size:12px;background:var(--jade);color:var(--card);padding:2px 6px;flex:none}
.mini-meta{margin:7px 0 0;font-family:var(--mono);font-size:11px;letter-spacing:.03em;text-transform:uppercase;color:var(--ink-soft)}
.mini-hours{margin:4px 0 0;font-size:13px;color:var(--ink-soft)}

/* pending */
.pending{display:flex;align-items:center;gap:11px;font-family:var(--mono);font-size:12px;letter-spacing:.06em;color:var(--jade)}
.pips{display:inline-flex;gap:4px}
.pip{width:6px;height:6px;background:var(--jade);animation:blink 1.1s infinite}
.pip:nth-child(2){animation-delay:.18s}
.pip:nth-child(3){animation-delay:.36s}
@keyframes blink{0%,60%,100%{opacity:.22}30%{opacity:1}}

/* error */
.err{background:var(--card);border:1px solid var(--chili);border-left:3px solid var(--chili);padding:14px 16px}
.err-head{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--chili);margin:0 0 6px}
.err p{margin:0;font-size:14px;line-height:1.5}

/* composer */
.composer{position:sticky;bottom:0;background:var(--paper);border-top:1px solid var(--line);padding:16px 24px 22px}
.composer-inner{max-width:760px;margin:0 auto;display:flex;gap:10px;align-items:flex-end}
.composer textarea{
  flex:1;resize:none;font:inherit;font-size:15px;line-height:1.5;
  background:var(--card);border:1px solid var(--line);padding:14px 16px;color:var(--ink);
  min-height:52px;font-family:var(--body);
}
.composer textarea:focus{border-color:var(--jade)}
.composer textarea::placeholder{color:var(--ink-soft)}
.send{
  background:var(--ink);color:var(--card);border:0;padding:0 22px;height:52px;flex:none;
  font-family:var(--display);font-weight:700;font-size:15px;
  transition:background .12s ease;
}
.send:hover:not(:disabled){background:var(--jade)}
.send:disabled{background:transparent;color:var(--ink-soft);border:1px dashed var(--line);cursor:not-allowed}
.hint{max-width:760px;margin:9px auto 0;font-family:var(--mono);font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-soft)}

@media (max-width:640px){
  .bar{padding:12px 16px}
  .thread,.composer{padding-left:16px;padding-right:16px}
  .msg-user .bubble{max-width:92%}
  .brand-name{font-size:15px}
}
@media (prefers-reduced-motion:reduce){
  .app *{transition:none!important;animation:none!important}
}
`}</style>

      <header className="bar">
        <div className="brand">
          <span className="brand-mark">Agent</span>
          <span className="brand-name">Restaurant Reservations</span>
        </div>
        {started && (
          <button className="reset" type="button" onClick={reset}>
            New request
          </button>
        )}
      </header>

      <div className="thread" ref={threadRef}>
        <div className="thread-inner">
          {!started && (
            <div className="opening">
              <p className="eyebrow">Reasoning + acting</p>
              <h1 className="title">
                Say where, when,
                <br />
                and how many.
              </h1>
              <p className="lede">
                Describe the meal in your own words. The agent searches restaurants, checks
                the tables, and books one — and shows every step it took.
              </p>
              <div className="examples">
                <p className="ex-label">Try one</p>
                {EXAMPLES.map((e) => (
                  <button key={e} className="ex" type="button" onClick={() => send(e)}>
                    <span className="ex-arrow">→</span>
                    <span>{e}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, i) =>
            m.role === "user" ? (
              <div key={i} className="msg msg-user">
                <span className="who">You</span>
                <div className="bubble">{m.text}</div>
              </div>
            ) : (
              <div key={i} className={`msg msg-agent${m.done ? " is-done" : ""}`}>
                <span className="who">Agent</span>
                <div className="bubble">
                  {m.text}
                  {m.done && <span className="confirm">Reservation confirmed</span>}
                  <ResultStrip restaurants={m.restaurants} />
                </div>
                <Trace steps={m.trace} />
              </div>
            )
          )}

          {pending && (
            <div className="pending">
              <span className="pips">
                <span className="pip" />
                <span className="pip" />
                <span className="pip" />
              </span>
              {pending}
            </div>
          )}

          {error && (
            <div className="err">
              <p className="err-head">Request failed</p>
              <p>{error}</p>
            </div>
          )}
        </div>
      </div>

      <div className="composer">
        <div className="composer-inner">
          <textarea
            ref={taRef}
            rows={1}
            value={input}
            placeholder="Table for two, Sichuan, Saturday around 8pm…"
            onChange={(e) => {
              setInput(e.target.value);
              grow();
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(input);
              }
            }}
          />
          <button
            className="send"
            type="button"
            disabled={!input.trim() || !!pending}
            onClick={() => send(input)}
          >
            Send
          </button>
        </div>
        <p className="hint">Enter to send · Shift + Enter for a new line</p>
      </div>
    </div>
  );
}
