interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface SearchStep {
  message: string;
  status: 'pending' | 'completed' | 'current';
}

interface ChatHistory {
  [chatId: string]: Message[];
}

interface SearchStepsHistory {
  [chatId: string]: SearchStep[];
}

class MessageStore {
  private static instance: MessageStore;
  private chatHistory: ChatHistory = {};
  private searchStepsHistory: SearchStepsHistory = {};

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

  public setSearchSteps(chatId: string, steps: SearchStep[]): void {
    this.searchStepsHistory[chatId] = steps;
  }

  public getSearchSteps(chatId: string): SearchStep[] {
    return this.searchStepsHistory[chatId] || [];
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