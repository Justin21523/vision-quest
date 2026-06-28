/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f5f7ff',
          100: '#ebf0fe',
          200: '#ced9fd',
          300: '#b1c2fc',
          400: '#7694fa',
          500: '#3b66f8',
          600: '#355ce0',
          700: '#2c4dba',
          800: '#233d95',
          900: '#1d327a',
        },
      },
    },
  },
  plugins: [],
}
