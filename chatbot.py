# STEP 1: Import required libraries

import tkinter as tk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# STEP 2: Training data 

training_sentences = [
    "hello",
    "hi",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "hello bot",
    "hi chatbot",

    "how are you",
    "how are you doing",
    "how is it going",
    "are you fine",
    "how do you feel",

    "what is your name",
    "your name",
    "who are you",
    "tell me your name",
    "what should I call you",

    "help",
    "can you help me",
    "what can you do",
    "how can you help",
    "what are your features",

    "bye",
    "goodbye",
    "see you",
    "exit",
    "quit",
    "thanks bye"
]

training_labels = [
    "greeting", "greeting", "greeting", "greeting",
    "greeting", "greeting", "greeting", "greeting", 
    "status", "status", "status", "status", "status",
    "name", "name", "name", "name", "name",
    "help", "help", "help", "help", "help",
    "bye", "bye", "bye", "bye", "bye", "bye"
]

# STEP 3: Vectorization

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_sentences)

# STEP 4: Train the ML model

model = MultinomialNB()
model.fit(X, training_labels)

# STEP 5: Responses

responses = {
    "greeting": "Hello 😊 How can I help you today?",
    "status": "I'm doing great! Thanks for asking 😄",
    "name": "I'm a simple chatbot built using Scikit-learn 🤖",
    "help": "I can chat with you, answer basic questions, and guide you 😊",
    "bye": "Goodbye! Have a great day 👋"
}

# STEP 6: Chatbot logic

def chatbot_response(user_input):
    user_input = user_input.lower().strip()
    input_vector = vectorizer.transform([user_input])
    intent = model.predict(input_vector)[0]
    return responses[intent]

# STEP 7: Tkinter GUI

def send_message():
    user_message = entry.get()
    if user_message == "":
        return

    chat_area.insert(tk.END, "You: " + user_message + "\n")
    reply = chatbot_response(user_message)
    chat_area.insert(tk.END, "Bot: " + reply + "\n\n")

    entry.delete(0, tk.END)

    if reply == responses["bye"]:
        window.after(1200, window.destroy)

window = tk.Tk()
window.title("ML Chatbot")
window.geometry("420x520")
window.config(bg="#E8F6F3") 

chat_area = tk.Text(
    window,
    font=("Arial", 11),
    bg="#FFFFFF",
    fg="#000000",
    bd=2,
    relief=tk.GROOVE,
    wrap=tk.WORD
)
chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

chat_area.tag_config("user", foreground="#1F618D", font=("Arial", 11, "bold"))
chat_area.tag_config("bot", foreground="#117864", font=("Arial", 11))
chat_area.insert(tk.END, "Bot: Hello! I'm your chatbot 😊\n\n", "bot")

input_frame = tk.Frame(window, bg="#E8F6F3")
input_frame.pack(fill=tk.X, padx=10, pady=5)


entry = tk.Entry(
    input_frame,
    font=("Arial", 12),
    bd=2,
    relief=tk.SOLID
)
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

send_button = tk.Button(
    input_frame,
    text="Send",
    font=("Arial", 12, "bold"),
    bg="#48C9B0",
    fg="white",
    activebackground="#1ABC9C",
    bd=0,
    padx=15,
    command=send_message
)
send_button.pack(side=tk.RIGHT)


window.mainloop()
