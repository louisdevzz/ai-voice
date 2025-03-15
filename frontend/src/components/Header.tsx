import React from 'react';
import { Copy, Download, Menu } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <header className="bg-white shadow-md fixed w-full top-0 z-50 border-none">
      <div className="w-full px-2 md:px-4 py-2 md:py-3 flex items-center justify-between border-none">
        <a href="/" className="flex items-center">
          <img 
            src="https://ttu.edu.vn/wp-content/uploads/2017/09/logo-ttu-41.png" 
            alt="Tan Tao University Logo" 
            className="h-8 md:h-12 w-auto"
          />
        </a>
        <div className="flex items-center space-x-2 md:space-x-4">
          <button className="p-1.5 md:p-2 hover:bg-gray-100 rounded-full">
            <Copy className="w-4 h-4 md:w-5 md:h-5 text-gray-600" />
          </button>
          <button className="p-1.5 md:p-2 hover:bg-gray-100 rounded-full">
            <Download className="w-4 h-4 md:w-5 md:h-5 text-gray-600" />
          </button>
          <button className="p-1.5 md:p-2 hover:bg-gray-100 rounded-full">
            <Menu className="w-4 h-4 md:w-5 md:h-5 text-gray-600" />
          </button>
          <button className="p-0.5 md:p-1 hover:ring-2 hover:ring-gray-300 rounded-full">
            <img 
              src="https://i.pinimg.com/736x/3d/96/ce/3d96ce6920a533a07ebb2770d307f1eb.jpg"
              alt="User Avatar"
              className="w-6 h-6 md:w-8 md:h-8 rounded-full"
            />
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header; 