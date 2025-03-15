import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Chat from './components/Chat';
import ChatPage from './components/ChatPage';
import Header from './components/Header';
function App() {
  return (
    <Router>
      <Header />
      <div className="min-h-screen bg-white">
        <Routes>
          <Route path="/" element={<Chat />} />
          <Route path="/chat/:chatId" element={<ChatPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
