---
date: 2022-04-02
authors: [hermann-web]
comments: true
title: "Introducing the `lissajou` Package: Animate Stunning Lissajou Curves and Beyond"
---

# Introducing the [lissajou Package](https://pypi.org/project/lissajou/): Animate Stunning Lissajou Curves and Beyond

[Lissajou curves](https://en.wikipedia.org/wiki/Lissajous_curve) have always held a certain fascination for me. Their intricate, ever-changing patterns hold a beauty that's both mathematical and artistic. So, one day, I decided to embark on a journey to bring these curves to life through animation.

**A Look Back: The Spark of Inspiration:**

It all started with a simple Wikipedia search. I was particularly drawn to the idea of closed sinusoids, those graceful loops that seemed to dance across the screen. As an educator who enjoys helping students visualize complex concepts, I saw an opportunity to explore the animation of functions changing over time, something I hadn't delved into deeply before.

Finding a solution for plotting functions dynamically with matplotlib wasn't too challenging. I stumbled upon some [helpful code online online](https://stackoverflow.com/a/68957469) that dealt with animating images. However, I wanted to adapt it to my specific needs. While the original code relied on global variables to maintain persistence across animation frames, I opted for a more structured approach using classes. This allowed for cleaner code organization and easier abstraction for generating various figures.

**The Exploration Begins: From Simple to Stunning:**

Through experimentation, I explored various shapes like sinusoids, ellipses, and circles. Adjusting the animation speed proved to be a crucial aspect. Sometimes, the dynamic view offered an exhilarating glimpse into the evolution of the curves. Other times, a static image, capturing the essence of a closed sinusoid, held its own charm.

**The Unexpected Delight: A Dynamic Ratio:**

The real magic happened when I decided to experiment with a dynamic ratio for the Lissajou curve parameters. Introducing `cos(t/100)` into the equation for the ratio `a/b` created a truly mesmerizing effect. The curves seemed to pulsate and transform with an added layer of complexity. This unexpected discovery was a highlight of the entire project for me.

**Sharing the Joy: Invitation to Play:**

I encourage you to explore the code and animations yourself. The beauty of open-source tools like `matplotlib` and the power of Python made this project a rewarding experience.

The good news? You don't need to be a coding expert to create stunning animations! Here's how to get started with the [Lissajou Animation Framework](https://github.com/Hermann-web/lissajou):

1. **Installation:** Open your terminal (command prompt) and type: `pip install lissajou`

2. **Import the Library:** In your Python script, type:

```python
from lissajou.anim import GenericLissajou
```

3. **Create an Animation:** Now, create an animation object:

```python
animation = GenericLissajou()
```

4. **Run the Animation:** Simply call the `show()` method to display your animation:

```python
animation.show()
```

That's it! This code will generate a basic Lissajou curve animation.

And like in a restaurant, you can update the parameters of the lissajou function as well as the visualiation configurations

Also, like in a restaurant, you have more a variety of proposotions: Instead of the  Generic Lissajou Curve (`GenericLissajou`), you can also test out Lissajou Curve with Varying Amplitude, with Sinusoidal Amplitude Lissajou Curve, with Fixed Ratio, Ellipse Animation and more.

You will find more informations on the [documentation online](https://pypi.org/project/lissajou/)

**Beyond the Basics: Explore the Full Potential:**

The Lissajou Animation Framework is much more than just Lissajou curves! It offers a powerful toolkit for creating diverse animations:

* **Image Animation:** Bring static images to life by dynamically changing their pixel values.
* **2D Parametric Animation:** Create animations based on mathematical equations, allowing for endless possibilities.
* **3D Parametric Animation (Optional Libraries Needed):** Take your animations to a new dimension by generating and visualizing 3D shapes in motion.

The framework also includes pre-implemented animations for various Lissajou curve types, saving you time and effort.

**Dive Deeper:**

For a comprehensive look at the framework's capabilities and advanced usage, check out the [pypi project](https://pypi.org/project/lissajou/) including [full documentation](https://github.com/Hermann-web/lissajou/blob/master/docs/index.md). But don't be afraid to experiment and explore on your own! The world of animation awaits.

Let's create something beautiful together!
