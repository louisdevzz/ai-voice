import React, { useState } from 'react';
import { Menu, X } from 'lucide-react';

const Header: React.FC = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

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
        
        {/* Mobile menu button */}
        <div className="flex items-center md:hidden">
          <button 
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            className="p-2 rounded-md text-gray-600 hover:text-gray-900 hover:bg-gray-100"
          >
            {isMenuOpen ? (
              <X className="h-6 w-6" />
            ) : (
              <Menu className="h-6 w-6" />
            )}
          </button>
        </div>

        {/* Desktop Navigation */}
        <div className="hidden md:flex items-center space-x-4">
          <a 
            href="https://ttu.edu.vn" 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
          >
            Trường Đại học Tân Tạo
          </a>
          <a 
            href="https://sit.ttu.edu.vn" 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
          >
            Khoa Công nghệ thông tin
          </a>
          <a 
            href="https://tuyensinh.ttu.edu.vn" 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
          >
            Tuyển sinh
          </a>
          <button className="p-0.5 hover:ring-2 hover:ring-gray-300 rounded-full">
            <img 
              src="https://lh3.googleusercontent.com/proxy/lnY42VpSrP0mTHE2geGgua4OPG6r8W8XjLDwQQ_mnlBSaeOracxo0dZ2L4v8V055DC-K_1kyZoBT3rNjyM91pLW--V9FnpU9Wfs6QHsPRIJ9KpKg07KVRtb6RNYM"
              alt="User Avatar"
              className="w-6 h-6 md:w-8 md:h-8 rounded-full"
            />
          </button>
        </div>
      </div>

      {/* Mobile Navigation */}
      {isMenuOpen && (
        <div className="md:hidden">
          <div className="px-2 pt-2 pb-3 space-y-1 bg-white border-t">
            <a 
              href="https://ttu.edu.vn" 
              target="_blank" 
              rel="noopener noreferrer"
              className="block px-3 py-2 rounded-md text-base font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-50"
            >
              Trường Đại học Tân Tạo
            </a>
            <a 
              href="https://sit.ttu.edu.vn" 
              target="_blank" 
              rel="noopener noreferrer"
              className="block px-3 py-2 rounded-md text-base font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-50"
            >
              Khoa Công nghệ thông tin
            </a>
            <a 
              href="https://tuyensinh.ttu.edu.vn" 
              target="_blank" 
              rel="noopener noreferrer"
              className="block px-3 py-2 rounded-md text-base font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-50"
            >
              Tuyển sinh
            </a>
          </div>
        </div>
      )}
    </header>
  );
};

export default Header; 