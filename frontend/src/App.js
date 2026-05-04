import { useState, useEffect, useRef } from "react";
import './App.css';

function App() {
  // messages state (starts with opening AI message)
  const [messages, setMessages] = useState([
    { type: "ai", content: "Hello! Which recipe are we making today?" }
  ]);

  // input state
  const [input, setInput] = useState("");

  // handle sending message
  const handleSend = async () => {
    if (input.trim() === "") return;

    const userMessage = { type: "user", content: input };

    // show user message immediately
    setMessages(prev => [...prev, userMessage]);

    setInput("");

    try {
      const res = await fetch("http://localhost:5000/api", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt: input }),
      });

      const data = await res.json();

      const aiMessage = {
        type: "ai",
        content: data.response,
      };

      setMessages(prev => [...prev, aiMessage]);

    } catch (error) {
      setMessages(prev => [
        ...prev,
        { type: "ai", content: "Error: could not reach AI." }
      ]);
    }
  };

  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="app">
    <h1 className="title">ChefGPT</h1>

    {/* Chat area */}
    <div className="chat-container">
      {messages.map((msg, index) => (
        <div key={index} className="message">
          <div className="label">
            {msg.type === "user" ? "You" : "ChefGPT"}:
          </div>

          <div className={`text ${msg.type}`}>
            {msg.content}
          </div>
        </div>
      ))}

      <div ref={bottomRef} />
    </div>

    {/* Input area */}
    <div className="input-container">
      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && handleSend()}
        placeholder="What are we cooking?"
      />
      <button onClick={handleSend}>Cook</button>
    </div>
  </div>
  );
}

export default App;