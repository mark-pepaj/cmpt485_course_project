import { useState } from "react";

function App() {
  // messages state (starts with opening AI message)
  const [messages, setMessages] = useState([
    { type: "ai", content: "Hello! Which recipe are we making today?" }
  ]);

  // input state
  const [input, setInput] = useState("");

  // handle sending message
  const handleSend = () => {
    if (input.trim() === "") return;

    const userMessage = { type: "user", content: input };

    // fake AI response (hardcoded)
    const fakeResponse = {
      type: "ai",
      content: `Chocolate Cake

      Ingredients:
      - 2 cups flour
      - 1 cup sugar
      - 2 eggs

      Directions:
      1. Mix ingredients
      2. Bake at 350°F for 30 minutes`
    };

    setMessages([...messages, userMessage, fakeResponse]);

    setInput("");
  };

  return (
    <div style={{ padding: "20px", maxWidth: "600px", margin: "auto" }}>
      <h1>Recipe AI</h1>

      {/* Chat area */}
      <div style={{ border: "1px solid #ccc", padding: "10px", minHeight: "300px" }}>
        {messages.map((msg, index) => (
          <div key={index} style={{ marginBottom: "10px" }}>
            <b>{msg.type === "user" ? "You" : "ChefGPT"}:</b>
            <div style={{ whiteSpace: "pre-line" }}>
              {msg.content}
            </div>
          </div>
        ))}
      </div>

      {/* Input area */}
      <div style={{ marginTop: "10px" }}>
        <input
          style={{ width: "70%", padding: "8px" }}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              handleSend();
            }
          }}
          placeholder="Enter a recipe..."
        />
        <button onClick={handleSend} style={{ padding: "8px", marginLeft: "5px" }}>
          Send
        </button>
      </div>
    </div>
  );
}

export default App;