import turtle


screen = turtle.Screen()
screen.bgcolor("black")
screen.title("Star")


t = turtle.Turtle()
t.speed(5)
t.color("yellow")


size = 100
for _ in range(5):
    t.forward(size)
    t.right(144)


t.hideturtle()
turtle.done()
