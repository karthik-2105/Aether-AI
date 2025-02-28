from manim import *

class CNN(Scene):
    def construct(self):
        title = Text("Convolutional Neural Network").scale(0.9)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        # Rectangles for CNN layers
        input_layer = Rectangle(height=3.0, width=1.5, color=BLUE)
        conv_layer1 = Rectangle(height=2.5, width=1.5, color=YELLOW)
        pooling_layer1 = Rectangle(height=2.0, width=1.5, color=GREEN)
        conv_layer2 = Rectangle(height=1.5, width=1.5, color=YELLOW)
        pooling_layer2 = Rectangle(height=1.0, width=1.5, color=GREEN)
        fc_layer = Rectangle(height=0.5, width=1.5, color=RED)

        # Labels for the layers
        input_text = Text("Input Layer").scale(0.5).next_to(input_layer, DOWN)
        conv_text1 = Text("Conv Layer 1").scale(0.5).next_to(conv_layer1, DOWN)
        pooling_text1 = Text("Pooling Layer 1").scale(0.5).next_to(pooling_layer1, DOWN)
        conv_text2 = Text("Conv Layer 2").scale(0.5).next_to(conv_layer2, DOWN)
        pooling_text2 = Text("Pooling Layer 2").scale(0.5).next_to(pooling_layer2, DOWN)
        fc_text = Text("Fully Connected").scale(0.5).next_to(fc_layer, DOWN)

        # Position layers
        layers = VGroup(input_layer, conv_layer1, pooling_layer1, conv_layer2, pooling_layer2, fc_layer)
        layers.arrange(RIGHT, buff=0.5)

        # Create layer groups
        inputs = Group(input_layer, input_text)
        conv1 = Group(conv_layer1, conv_text1)
        pool1 = Group(pooling_layer1, pooling_text1)
        conv2 = Group(conv_layer2, conv_text2)
        pool2 = Group(pooling_layer2, pooling_text2)
        fc = Group(fc_layer, fc_text)

        # Add all groups to the scene
        self.play(FadeIn(inputs))
        self.wait(0.5)
        self.play(TransformFromCopy(inputs, conv1))
        self.wait(0.5)
        self.play(TransformFromCopy(conv1, pool1))
        self.wait(0.5)
        self.play(TransformFromCopy(pool1, conv2))
        self.wait(0.5)
        self.play(TransformFromCopy(conv2, pool2))
        self.wait(0.5)
        self.play(TransformFromCopy(pool2, fc))
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)))

# To display this animation, create an appropriate configuration file or execute it in a compatible environment.