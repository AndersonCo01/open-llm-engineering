import { useDeferredValue, useState } from "react";
import { categories, concepts, type Category } from "./data/concepts";
import { ArrowIcon, BookIcon, BookmarkIcon, ChartIcon, CheckIcon, FlaskIcon, QuizIcon } from "./components/Icons";

type Tab = "learn" | "library" | "quiz" | "progress";
type SavedState = { cardIndex: number; bookmarks: string[]; correct: number; answered: number };

const STORAGE_KEY = "llm-pocket-lab-progress-v1";

function readSavedState(): SavedState {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved ? JSON.parse(saved) : { cardIndex: 0, bookmarks: [], correct: 0, answered: 0 };
  } catch {
    return { cardIndex: 0, bookmarks: [], correct: 0, answered: 0 };
  }
}

function App() {
  const [initial] = useState(readSavedState);
  const [tab, setTab] = useState<Tab>("learn");
  const [cardIndex, setCardIndex] = useState(initial.cardIndex % concepts.length);
  const [bookmarks, setBookmarks] = useState<string[]>(initial.bookmarks);
  const [detailsOpen, setDetailsOpen] = useState(true);
  const [quizIndex, setQuizIndex] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);
  const [correct, setCorrect] = useState(initial.correct);
  const [answered, setAnswered] = useState(initial.answered);

  const persist = (next: Partial<SavedState>) => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ cardIndex, bookmarks, correct, answered, ...next }));
  };

  const changeCard = (next: number) => {
    const normalized = (next + concepts.length) % concepts.length;
    setCardIndex(normalized);
    setDetailsOpen(true);
    persist({ cardIndex: normalized });
  };

  const toggleBookmark = () => {
    const id = concepts[cardIndex].id;
    const next = bookmarks.includes(id) ? bookmarks.filter((item) => item !== id) : [...bookmarks, id];
    setBookmarks(next);
    persist({ bookmarks: next });
  };

  const selectAnswer = (index: number) => {
    if (selected !== null) return;
    const isCorrect = index === concepts[quizIndex].answer;
    const nextCorrect = correct + (isCorrect ? 1 : 0);
    const nextAnswered = answered + 1;
    setSelected(index);
    setCorrect(nextCorrect);
    setAnswered(nextAnswered);
    persist({ correct: nextCorrect, answered: nextAnswered });
  };

  const nextQuestion = () => {
    setQuizIndex((current) => (current + 1) % concepts.length);
    setSelected(null);
  };

  return (
    <main className="app-shell">
      <section className="phone-surface">
        {tab === "learn" ? (
          <LearnView cardIndex={cardIndex} bookmarks={bookmarks} detailsOpen={detailsOpen} onDetails={() => setDetailsOpen((open) => !open)} onBookmark={toggleBookmark} onChange={changeCard} />
        ) : null}
        {tab === "quiz" ? (
          <QuizView quizIndex={quizIndex} selected={selected} onSelect={selectAnswer} onNext={nextQuestion} />
        ) : null}
        {tab === "library" ? (
          <LibraryView onStudy={(index) => { setCardIndex(index); setTab("learn"); }} />
        ) : null}
        {tab === "progress" ? (
          <ProgressView correct={correct} answered={answered} bookmarks={bookmarks} onStudy={(index) => { setCardIndex(index); setTab("learn"); }} />
        ) : null}
        <BottomNav active={tab} onChange={setTab} />
      </section>
    </main>
  );
}

function LibraryView({ onStudy }: { onStudy: (index: number) => void }) {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState<Category | "All">("All");
  const deferredQuery = useDeferredValue(query.trim().toLowerCase());
  const matches = concepts
    .map((concept, index) => ({ concept, index }))
    .filter(({ concept }) => {
      const matchesCategory = category === "All" || concept.category === category;
      const searchable = `${concept.title} ${concept.definition} ${concept.code}`.toLowerCase();
      return matchesCategory && searchable.includes(deferredQuery);
    });

  return (
    <div className="view library-view">
      <header className="library-header">
        <h1>Concept library</h1>
        <p>Search the terminology and syntax used throughout the project.</p>
      </header>
      <div className="library-filters">
        <label>
          <span>Search terms</span>
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="attention, tensor, class" type="search" />
        </label>
        <label>
          <span>Category</span>
          <select value={category} onChange={(event) => setCategory(event.target.value as Category | "All")}>
            <option value="All">All categories</option>
            {categories.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </label>
      </div>
      <div className="library-summary"><strong>{matches.length}</strong> of {concepts.length} terms</div>
      {matches.length ? (
        <div className="term-list">
          {matches.map(({ concept, index }) => (
            <button key={concept.id} onClick={() => onStudy(index)}>
              <span><small>{concept.category}</small><strong>{concept.title}</strong><em>{concept.definition}</em></span>
              <ArrowIcon size={20} />
            </button>
          ))}
        </div>
      ) : <div className="empty-state"><p>No terms match that search. Try another word or category.</p></div>}
    </div>
  );
}

function Header({ cardIndex }: { cardIndex: number }) {
  const percent = ((cardIndex + 1) / concepts.length) * 100;
  return (
    <header className="topbar">
      <div className="brand"><FlaskIcon size={30} /><span>LLM Pocket Lab</span></div>
      <div className="progress-ring" style={{ "--progress": `${percent * 3.6}deg` } as React.CSSProperties} aria-label={`Card ${cardIndex + 1} of ${concepts.length}`}>
        <span>{cardIndex + 1} / {concepts.length}</span>
      </div>
    </header>
  );
}

type LearnProps = { cardIndex: number; bookmarks: string[]; detailsOpen: boolean; onDetails: () => void; onBookmark: () => void; onChange: (index: number) => void };
function LearnView({ cardIndex, bookmarks, detailsOpen, onDetails, onBookmark, onChange }: LearnProps) {
  const concept = concepts[cardIndex];
  return (
    <div className="view learn-view">
      <Header cardIndex={cardIndex} />
      <div className="category-row"><span>{concept.category}</span><span>Card {cardIndex + 1}</span></div>
      <article className="learning-card">
        <div className="card-title-row">
          <h1>{concept.title}</h1>
          <button className="icon-button" onClick={onBookmark} aria-label={bookmarks.includes(concept.id) ? "Remove bookmark" : "Bookmark card"}><BookmarkIcon filled={bookmarks.includes(concept.id)} /></button>
        </div>
        <p className="definition">{concept.definition}</p>
        <div className="divider" />
        <p className="field-label">Example</p>
        <pre className="code-box"><code>{concept.code}</code></pre>
        <p className="field-label">Output</p>
        <pre className="output-box"><code>{concept.output}</code></pre>
        <button className="details-toggle" onClick={onDetails} aria-expanded={detailsOpen}><span className={detailsOpen ? "chevron open" : "chevron"}>⌄</span>How it works</button>
        {detailsOpen ? <ol className="steps">{concept.steps.map((step) => <li key={step}>{step}</li>)}</ol> : null}
      </article>
      <div className="card-controls">
        <button className="nav-arrow secondary" onClick={() => onChange(cardIndex - 1)} aria-label="Previous card"><ArrowIcon direction="left" /></button>
        <div className="dots" aria-hidden="true">{concepts.slice(0, 7).map((item, index) => <span key={item.id} className={index === cardIndex % 7 ? "active" : ""} />)}</div>
        <button className="nav-arrow primary" onClick={() => onChange(cardIndex + 1)} aria-label="Next card"><ArrowIcon /></button>
      </div>
    </div>
  );
}

type QuizProps = { quizIndex: number; selected: number | null; onSelect: (index: number) => void; onNext: () => void };
function QuizView({ quizIndex, selected, onSelect, onNext }: QuizProps) {
  const concept = concepts[quizIndex];
  return (
    <div className="view quiz-view">
      <header className="quiz-header"><h1>Quick check</h1><p>Question {(quizIndex % 5) + 1} of 5</p></header>
      <div className="quiz-progress">{[0, 1, 2, 3, 4].map((step) => <span key={step} className={step <= quizIndex % 5 ? "active" : ""} />)}</div>
      <section className="quiz-content">
        <p className="quiz-category">{concept.category}</p>
        <h2>{concept.question}</h2>
        <div className="answers">{concept.options.map((option, index) => {
          const isChosen = selected === index;
          const isCorrect = selected !== null && index === concept.answer;
          const isWrong = isChosen && !isCorrect;
          return <button key={option} className={`answer ${isCorrect ? "correct" : ""} ${isWrong ? "wrong" : ""}`} onClick={() => onSelect(index)} disabled={selected !== null}><span className="radio">{isCorrect ? <CheckIcon size={18} /> : null}</span><span>{option}</span></button>;
        })}</div>
        {selected !== null ? <div className={`explanation ${selected === concept.answer ? "success" : "retry"}`}><strong>{selected === concept.answer ? "Correct" : "Keep learning"}</strong><p>{concept.explanation}</p></div> : <div className="quiz-hint">Choose the best answer. You’ll get an explanation immediately.</div>}
      </section>
      <button className="next-button" onClick={onNext} disabled={selected === null}>Next question</button>
    </div>
  );
}

type ProgressProps = { correct: number; answered: number; bookmarks: string[]; onStudy: (index: number) => void };
function ProgressView({ correct, answered, bookmarks, onStudy }: ProgressProps) {
  const score = answered ? Math.round((correct / answered) * 100) : 0;
  const savedConcepts = concepts.map((concept, index) => ({ concept, index })).filter(({ concept }) => bookmarks.includes(concept.id));
  return (
    <div className="view progress-view">
      <header className="progress-header"><h1>Your progress</h1><p>Small steps turn into strong foundations.</p></header>
      <section className="score-panel"><div className="score-circle"><strong>{score}%</strong><span>quiz score</span></div><div><strong>{correct} correct</strong><span>from {answered} answered</span></div></section>
      <section className="progress-section"><h2>Course map</h2><div className="course-track"><div className="course-fill" style={{ width: `${(2 / 14) * 100}%` }} /></div><div className="course-labels"><span>2 days complete</span><span>14 total</span></div></section>
      <section className="progress-section"><h2>Saved cards</h2>{savedConcepts.length ? <div className="saved-list">{savedConcepts.map(({ concept, index }) => <button key={concept.id} onClick={() => onStudy(index)}><span><small>{concept.category}</small>{concept.title}</span><ArrowIcon size={20} /></button>)}</div> : <div className="empty-state"><BookmarkIcon /><p>Bookmark a card and it will appear here.</p></div>}</section>
    </div>
  );
}

function BottomNav({ active, onChange }: { active: Tab; onChange: (tab: Tab) => void }) {
  const items: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: "learn", label: "Learn", icon: <BookIcon /> },
    { id: "library", label: "Library", icon: <FlaskIcon /> },
    { id: "quiz", label: "Quiz", icon: <QuizIcon /> },
    { id: "progress", label: "Progress", icon: <ChartIcon /> },
  ];
  return <nav className="bottom-nav" aria-label="Primary navigation">{items.map((item) => <button key={item.id} className={active === item.id ? "active" : ""} onClick={() => onChange(item.id)}>{item.icon}<span>{item.label}</span></button>)}</nav>;
}

export default App;
