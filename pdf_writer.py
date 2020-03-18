from reportlab.pdfgen import canvas


'''
    PDFWriter
    Write schedules to PDF files
'''
class PDFWriter():
    
    def __init__(self):
        
        self.POINT = 1
        self.INCH = 72
        self.FONT_SIZE = 12
        self.COL_START = [1 * self.INCH,
                          3.0 * self.INCH,
                          5.5 * self.INCH]
        self.RECTANGLE_END = (8.5 - 1) * self.INCH
        self.TEXT_HEIGHT = 14 * self.POINT

        
    def make_pdf_file(self, output_filename, text, chart_lines):
        
        c = canvas.Canvas(output_filename, pagesize=(8.5 * self.INCH, 11 * self.INCH))
        c.setStrokeColorRGB(0,0,0)
        c.setFillColorRGB(0,0,0)
        c.setFont("Helvetica", self.FONT_SIZE * self.POINT)
        v = 10 * self.INCH
        for line_num, line in enumerate(text):
            
            # Hightlight alternating lines
            if (line_num >= chart_lines[0] and
                line_num <= chart_lines[1]):
                
                if (line_num - chart_lines[0]) % 2 == 0:
                    c.setFillColorRGB(0.9, 0.9, 0.9) 
                    c.rect(self.COL_START[0],
                           v - 2 * self.POINT,
                           self.RECTANGLE_END - self.COL_START[0],
                           self.TEXT_HEIGHT,
                           stroke=0,
                           fill=1)
            
            # Write the text
            c.setFillColorRGB(0, 0, 0)
            for col_num, col in enumerate(line):
                string = self.clean_string(col)
                c.drawString(self.COL_START[col_num], v, string)
                
            # Find the height of the next line of text
            v -= self.TEXT_HEIGHT
            
        # Create the file
        c.showPage()
        c.save()
        
    def clean_string(self, string):
        
        string = string.replace('[', '')
        string = string.replace(']', '')
        string = string.replace("'", '')
        
        return string