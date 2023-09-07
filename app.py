from flask import Flask, request, jsonify, render_template
import openai
from transformers import pipeline
import spacy

nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

# Set up OpenAI API key
openai.api_key = "ABC"

# Set up a sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2-english")

# List of subtopics
subtopics = ["Intro to ML", "Steps in ML Training", "Data Collection", "Preprocessing", "Training", "Evaluation"]

negative_words = ['no', 'not', 'nope', 'nay', 'not exact', 'not correct', 'incorrect', 'wrong', 'nah', "ain't", "don't", "sorry", "confused", "clear", 'misunderstanding']

# Initialize conversation
current_subtopic_index = 0
current_subtopic = subtopics[current_subtopic_index]
conversation_history = []

# Function to generate a response from ChatGPT
def generate_response(prompt, role1, role2):
    conversation_history.append({"role": role1, "content": prompt})
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=conversation_history)
    message = response['choices'][0]['message']['content'].strip()
    conversation_history.append({"role": role2, "content": message})
    return message

# Function to check if user's answer is correct
def check_answer(user_answer, question):
    prompt = f"Is '{user_answer}' a correct answer for '{question}'?"
    feedback = generate_response(prompt, "user", "assistant")

    doc = nlp(feedback)

    # Pass the first two sentences of the chatbot's response to the sentiment analysis pipeline
    first_two_sentences = list(doc.sents)[:2]
    flattened_text = ' '.join([sent.text for sent in first_two_sentences])
    sent = sentiment_analyzer(flattened_text)

    # Get the sentiment score
    score = sent[0]['score']

    if score < 0.5 or any(word in str(next(doc.sents)).lower() for word in negative_words) and "not bad" not in flattened_text.lower() and "not just" not in flattened_text.lower():
        return False, feedback
    else:
        return True, feedback
    
def add_space(text):
    # Replace every "." with ". "
    text_with_space = text.replace(".", ". ")
    return text_with_space
    
@app.route("/")
def home():
    return render_template("index.html")

# Endpoint to start conversation and get initial subtopic explanation
@app.route("/", methods=["POST"])
def start_conversation():
    global current_subtopic_index, current_subtopic, conversation_history
    conversation_history = [{"role": "system", "content": "You are an AI/ML tutor who speaks in the style of a mafia lord from a 1920s noir film. Limit your answers to 200 words."},
            {"role": "user", "content": "Teach me about intro to ML and ask a follow up question?"},
            {"role": "assistant", "content": "Alright, listen up. Machine learning is like having a whole crew of wise guys working for you around the clock. It's all about teaching machines to learn from data and make predictions on their own. With this tech, we can analyze vast amounts of data and identify new opportunities in our operations. It's the future, and it's gonna make us richer and more powerful than ever before. But let me tell ya, we gotta be smart about it. We gotta know what kinda data we're dealing with and how to use it to our advantage. We gotta keep our eyes peeled for any risks or pitfalls. And we gotta be careful not to rely on the machines too much - after all, there's no substitute for good old-fashioned street smarts. \n\nSo tell me, how can we use this ML to gain an edge in the game?"}
            ]
    prompt = f"Teach me about {current_subtopic} and ask a relevant follow up question."
    response = generate_response(prompt, "user", "assistant")
    doc = nlp(response)
    followup_question = ''.join([sent.text for sent in doc.sents if '?' in sent.text])
    response = ''.join([sent.text for sent in doc.sents if '?' not in sent.text])
    return jsonify({"message": add_space(response), "followup_question": followup_question})

# Endpoint to handle user response and provide feedback
@app.route("/response", methods=["POST"])
def handle_response():
    global current_subtopic_index, current_subtopic, conversation_history
    user_answer = request.json["answer"]
    conversation_history.append({"role": "user", "content": user_answer})
    is_correct = check_answer(user_answer, request.json['followup_question'])
    conversation_history.append({"role": "assistant", "content": is_correct[1]})
    if is_correct[0]:
        current_subtopic_index += 1
        if current_subtopic_index > len(subtopics):
            return jsonify({"message": "You have completed all the subtopics!", "followup_question": None})
        current_subtopic = subtopics[current_subtopic_index]
        status = f"Correct!\nOnto the next subtopic — {current_subtopic}"
        prompt = f"Teach me about {current_subtopic} and ask a relevant follow up question."
        response = generate_response(prompt, "user", "assistant")
        doc = nlp(response)
        followup_question = ''.join([sent.text for sent in doc.sents if '?' in sent.text])
        response = ''.join([sent.text for sent in doc.sents if '?' not in sent.text])
        return jsonify({"message": add_space(response), "followup_question": followup_question, "feedback": is_correct[1], "status": status})
    else:
        status = f"Looks like you're missing some details!\nLet's try {current_subtopic} again."
        prompt = f"Teach me about {current_subtopic} differently and ask a relevant followup question."
        message = generate_response(prompt, "user", "assistant")
        conversation_history.append({"role": "assistant", "content": message})
        doc = nlp(message)
        followup_question = ''.join([sent.text for sent in doc.sents if '?' in sent.text])
        message = ''.join([sent.text for sent in doc.sents if '?' not in sent.text])
        return jsonify({"message": add_space(message), "followup_question": followup_question, "feedback": is_correct[1], "status": status})

if __name__ == "__main__":
    app.run(debug=True)
