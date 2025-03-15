interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface ChatHistory {
  [chatId: string]: Message[];
}

class MessageStore {
  private static instance: MessageStore;
  private chatHistory: ChatHistory = {};

  private constructor() {}

  public static getInstance(): MessageStore {
    if (!MessageStore.instance) {
      MessageStore.instance = new MessageStore();
    }
    return MessageStore.instance;
  }

  public addMessage(chatId: string, message: Message): void {
    if (!this.chatHistory[chatId]) {
      this.chatHistory[chatId] = [];
    }
    this.chatHistory[chatId].push(message);
  }

  public getMessages(chatId: string): Message[] {
    return this.chatHistory[chatId] || [];
  }

  public addInitialMessage(chatId: string, userMessage: string, assistantMessage: string): void {
    if (!this.chatHistory[chatId]) {
      this.chatHistory[chatId] = [];
    }
    this.chatHistory[chatId].push(
      { role: 'user', content: userMessage },
      { role: 'assistant', content: assistantMessage }
    );
  }
}

export default MessageStore; 