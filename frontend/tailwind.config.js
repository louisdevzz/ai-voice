/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          DEFAULT: '#1E1E1E',
          '100': '#2A2A2A',
          '200': '#3A3A3A',
        }
      },
    },
  },
  plugins: [],
} 