from manim import *

class PingalaScene(Scene):
    def construct(self):
        title = Text("Pingala", color=BLUE).scale(1.5).to_edge(UP)
        
        # Lines of description
        line1 = Text("Pingala was an ancient Indian scholar known for his work", font_size=36)
        line2 = Text("in the field of combinatorial mathematics, specifically", font_size=36)
        line3 = Text("in the area of binary numeral systems.", font_size=36)
        
        # Grouping text lines
        description = VGroup(line1, line2, line3).arrange(DOWN).next_to(title, DOWN, buff=0.5)

        # Animation: display title, then one by one text lines.
        self.play(Write(title))
        self.play(FadeIn(description, shift=UP))
        self.wait()