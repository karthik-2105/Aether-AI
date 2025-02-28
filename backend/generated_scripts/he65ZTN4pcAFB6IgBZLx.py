from manim import *

class DNAStructure(Scene):
    def construct(self):
        # Write the title
        title = Text("DNA Structure").scale(1.5).shift(UP*3)
        self.play(Write(title))
        
        # Creating a basic DNA Structure
        left_helix = Line(ORIGIN, 2*UP).shift(LEFT*2)
        right_helix = Line(ORIGIN, 2*UP).shift(RIGHT*2)
        self.play(Create(left_helix), Create(right_helix))
        
        # Building blocks
        bases = VGroup()
        for i in range(5):
            base = Line(LEFT, RIGHT).set_color(WHITE).shift(UP*i*0.5)
            bases.add(base)
        
        connections = VGroup()
        for i in range(5):
            connection = DashedLine(left_helix.point_from_proportion(i*0.5/4),
                                    right_helix.point_from_proportion(i*0.5/4),
                                    dash_length=0.1, dashed_ratio=0.6).scale(0.2)
            connections.add(connection)
        
        self.play(Create(connections), run_time=2)
        self.play(Create(bases))
        
        # Adding text descriptions
        description1 = Text("Double Helix Structure").next_to(left_helix, RIGHT*3)
        description2 = Text("Base Pairs").next_to(bases, RIGHT*3)
        
        self.play(Write(description1))
        self.play(Write(description2))
        
        # Rotate DNA to show 3D aspect
        self.play(Rotate(connections, angle=TAU/4, axis=RIGHT), Rotate(bases, angle=TAU/4, axis=RIGHT), Rotate(left_helix, angle=TAU/4, axis=RIGHT), Rotate(right_helix, angle=TAU/4, axis=RIGHT), run_time=3)
        
        # More detailed description
        detail = Text("DNA consists of two long strands of nucleotides twisted into a double helix and joined by hydrogen bonds between the complementary bases adenine with thymine and cytosine with guanine.").scale(0.5).next_to(connections, DOWN*2)
        self.play(Write(detail))
        
        # Ending scene
        self.wait(2)
        self.play(FadeOut(detail), FadeOut(description1), FadeOut(description2), FadeOut(bases), FadeOut(connections), FadeOut(left_helix), FadeOut(right_helix), FadeOut(title))