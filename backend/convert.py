import PyPDF2
import os

def convert_pdf_to_text(pdf_path):
    try:
        # Check if file exists
        if not os.path.exists(pdf_path):
            print(f"Error: File '{pdf_path}' not found.")
            return None

        # Open the PDF file in binary read mode
        with open(pdf_path, 'rb') as file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Get total number of pages
            num_pages = len(pdf_reader.pages)
            print(f"Total number of pages: {num_pages}")
            
            # Extract text from all pages
            text = ""
            for page_num in range(num_pages):
                # Get the page object
                page = pdf_reader.pages[page_num]
                # Extract text from page
                text += page.extract_text()
                
            # Create output text file name
            output_file = pdf_path.rsplit('.', 1)[0] + '.txt'
            
            # Write text to file
            with open(output_file, 'w', encoding='utf-8') as text_file:
                text_file.write(text)
                
            print(f"Text has been extracted and saved to: {output_file}")
            return text
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    pdf_file = "Ver2. Đại học Tân Tạo Tuyển sinh 2025.pdf"
    text_content = convert_pdf_to_text(pdf_file)
