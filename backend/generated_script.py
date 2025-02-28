from manim import *

class ArrayInsertion(Scene):
    def construct(self):
        # Title
        title = Tex("Data Structures: Arrays").scale(1.2)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))
        
        # Introduction to Array
        array_intro = Tex(
            "An array is a collection of items\\\\",
            "stored at contiguous memory locations."
        ).scale(0.75)
        
        self.play(Write(array_intro))
        self.wait(3)
        self.play(FadeOut(array_intro))
        
        # Array Representation
        array = VGroup(*[Square().scale(0.5) for _ in range(5)])
        array.arrange(RIGHT, buff=0.1)
        array_numbers = VGroup(*[Tex(str(i)) for i in range(5)])
        for square, num in zip(array, array_numbers):
            num.move_to(square.get_center())
        
        self.play(FadeIn(array), run_time=1)
        self.play(Write(array_numbers), run_time=1)
        self.wait(2)
        
        # Insertion Animation
        insert_number = Tex("Insert 7 at index 2").to_edge(UP)
        self.play(Write(insert_number))
        
        new_square = Square().scale(0.5)
        new_num = Tex("7").scale(0.75).move_to(new_square.get_center())
        
        self.play(
            array[2:].animate.shift(RIGHT*0.6),
            array_numbers[2:].animate.shift(RIGHT*0.6),
        )
        self.wait(1)
        
        new_square.move_to(array[1].get_center() + 0.6 * RIGHT)
        self.play(FadeIn(new_square))
        self.play(Write(new_num))
        self.wait(2)
        
        self.play(FadeOut(array), FadeOut(array_numbers), FadeOut(new_square),
                  FadeOut(new_num), FadeOut(insert_number))

        # Wrap up the scene
        thank_you = Tex("Thank you for watching!").scale(1)
        self.play(Write(thank_you))
        self.wait(2)
        self.play(FadeOut(thank_you))
        
if __name__ == "__main__":
    scene = ArrayInsertion()
    scene.render()