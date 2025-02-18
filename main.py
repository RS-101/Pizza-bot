from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import ollama
import httpx

# Boolean flag to toggle between Gemini API and Ollama
USE_GEMINI = True

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyAj5Q6mwmZVCFZwyHEg2i3Z0xZKuF3rJpY"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

# FastAPI App
app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class MessageRequest(BaseModel):
    message: str
    use_gemini: bool = USE_GEMINI

SYSTEM_PROMPT = """
You are an intent classifier. Given a user input, classify it into one of the following intents:
- discount
- escalate
- exit
- greetings
- help
- order
- reset
- toppings_list
- fallback (if it doesn't match any intent)
"""

async def classify_intent_ollama(user_input: str) -> str:
    """Classifies intent using Ollama."""
    response = ollama.chat(
        model="gemma2:2b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]
    )
    return response['message']['content'].strip().lower()

async def classify_intent_gemini(user_input: str) -> str:
    """Classifies intent using Gemini API."""
    payload = {
        "contents": [{
            "parts": [{"text": f"{SYSTEM_PROMPT}\nUser input: {user_input}"}]
        }]
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(GEMINI_API_URL, json=payload)
        response_data = response.json()
    
    try:
        intent = response_data['candidates'][0]['content']['parts'][0]['text'].strip().lower()
    except (KeyError, IndexError, TypeError):
        intent = "fallback"
    
    return intent

async def classify_intent(user_input: str, use_gemini: bool) -> str:
    """Determines which intent classification method to use."""
    if use_gemini:
        return await classify_intent_gemini(user_input)
    return await classify_intent_ollama(user_input)

class ChatNode:
    def __init__(self, name):
        self.name = name
        self.intents = {}

    def add_intent(self, intent, response):
        self.intents[intent] = response

    def get_response(self, intent):
        return self.intents.get(intent, None)

class ChatBot:
    def __init__(self):
        # Mapping for non-order intents
        self.nodes = {}
        # Conversation context for an ongoing order
        self.conversation = {}
        # Order flow state: None, awaiting_name, awaiting_size, awaiting_toppings
        self.order_state = None

    def add_node(self, node):
        self.nodes[node.name] = node

    def handle_order_flow(self, user_input: str, intent: str = None) -> str:
        """Handles the ordering conversation flow.
           If an interruption intent (like 'toppings_list') is detected while waiting for toppings,
           the bot answers that query and then re-prompts for toppings.
        """
        if self.order_state is None:
            # Start the ordering process
            self.conversation = {}
            self.order_state = "awaiting_name"
            return "Let's start your order. What is your name?"
        elif self.order_state == "awaiting_name":
            self.conversation["name"] = user_input.strip()
            self.order_state = "awaiting_size"
            return f"Nice to meet you, {self.conversation['name']}! What pizza size would you like? (small, medium, large)"
        elif self.order_state == "awaiting_size":
            self.conversation["pizza_size"] = user_input.strip()
            self.order_state = "awaiting_toppings"
            return (f"Great! You've chosen a {self.conversation['pizza_size']} pizza. "
                    "What toppings would you like? Please list them separated by commas.")
        elif self.order_state == "awaiting_toppings":
            # Check if the user is asking an interruption question about toppings
            if intent == "toppings_list":
                toppings_info = self.nodes.get("toppings_list").get_response("toppings_list")
                followup = "Now, please list the toppings you'd like, separated by commas."
                return f"{toppings_info}\n{followup}"
            else:
                toppings = [t.strip() for t in user_input.split(",") if t.strip()]
                self.conversation["toppings"] = toppings
                confirmation = (
                    f"Thank you {self.conversation.get('name', 'customer')}! "
                    f"You have ordered a {self.conversation.get('pizza_size', '')} pizza with toppings: {', '.join(toppings)}."
                )
                # Reset order state after confirmation
                self.order_state = None
                self.conversation = {}
                return confirmation
        else:
            # Fallback if state is unknown
            self.order_state = None
            self.conversation = {}
            return "I'm sorry, something went wrong with your order. Let's start over. What is your name?"

    async def handle_input(self, user_input, use_gemini):
        intent = await classify_intent(user_input, use_gemini)
        print(f"Intent: {intent}")

        # If an order is in progress, handle the order flow with potential interruption.
        if self.order_state is not None or intent == "order":
            return self.handle_order_flow(user_input, intent)

        # Process non-order intents normally.
        if intent in self.nodes:
            response = self.nodes[intent].get_response(intent)
            return response

        print(f"No predefined response for intent '{intent}'. Querying AI...")
        # Fallback: query AI directly if no mapping is found.
        if use_gemini:
            payload = {"contents": [{"parts": [{"text": user_input}]}]}
            async with httpx.AsyncClient() as client:
                ai_response = await client.post(GEMINI_API_URL, json=payload)
                response_data = ai_response.json()
                try:
                    return response_data['candidates'][0]['content']['parts'][0]['text']
                except (KeyError, IndexError, TypeError):
                    return "I'm sorry, I couldn't process that request."
        else:
            ai_response = ollama.chat(model='gemma2:2b', messages=[{'role': 'user', 'content': user_input}])
            return ai_response['message']['content']

# Initialize chatbot and define nodes for other intents

bot = ChatBot()

# Node for "greetings" intent
greetings_node = ChatNode("greetings")
greetings_node.add_intent("greetings", "Welcome to Pizza Topping Basic demonstration. You can order a pizza with selected sizes, types, and toppings. Ask for help if needed.")

# Node for "help" intent
help_node = ChatNode("help")
help_node.add_intent("help", "Sure, I can help! If you want to order a pizza, just type 'order'.")

# Node for "discount" intent
discount_node = ChatNode("discount")
discount_node.add_intent("discount", "Yay!, you get a discount rebate.")

# Node for "escalate" intent
escalate_node = ChatNode("escalate")
escalate_node.add_intent("escalate", "Ok, let me connect you with a pizza expert.")

# Node for "exit" intent
exit_node = ChatNode("exit")
exit_node.add_intent("exit", "Goodbye! Have a great day.")

# Node for "reset" intent
reset_node = ChatNode("reset")
reset_node.add_intent("reset", "Let's start over.")

# Node for "toppings_list" intent
toppings_node = ChatNode("toppings_list")
toppings_node.add_intent("toppings_list", "We have cheese, sausage, and pepperoni.")

# Add nodes to the bot
bot.add_node(greetings_node)
bot.add_node(help_node)
bot.add_node(discount_node)
bot.add_node(escalate_node)
bot.add_node(exit_node)
bot.add_node(reset_node)
bot.add_node(toppings_node)

@app.post("/api/data")
async def get_data(request: MessageRequest):
    user_input = request.message.strip()
    response = await bot.handle_input(user_input, request.use_gemini)
    print(f"Response: {response}")
    return {"message": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
