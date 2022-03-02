import os
import sys
import pygame
from pygame import Surface
from pygame.locals import *

import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg

class Visualizer:

    def __init__(self, name='render', width=1000, height=1000):

        self.no_open_window = True
        self.name = name
        self.window_width = width
        self.window_height = height

        # Initialize pygame
        pygame.init()
        self.plot_surface = Surface([width, height])

        # Define some colors
        self.color_dict = {
            'no_items_white': (240, 240, 240),
            'white': (255, 255, 255),
            'light-grey': (195, 195, 195),
            'grey': (128, 128, 128),
            'black': (0, 0, 0),
            'half_transp': (255, 255, 255, 125),
            'full_transp': (255, 255, 255, 0),
            'red': (165, 36, 36),
            'green': (67, 149, 64),
            'blue': (81, 73, 186),
            'purple': (151, 69, 176),
            'light-blue': (65, 163, 212),
            'orange': (239, 179, 110),
            'yellow': (239, 203, 24),
        }

    def draw_plot(self, fig):
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        self.plot_surface = pygame.image.fromstring(raw_data, size, "RGB")

    def reset_surfaces(self):
        self.screen.fill(self.color_dict['white'])
        self.plot_surface.fill(self.color_dict['full_transp'])

    def render_step(self, fig):
        if self.no_open_window:
            self.screen = pygame.display.set_mode([self.window_width, self.window_height])
            pygame.display.set_caption(self.name)
            self.no_open_window = False

        self.reset_surfaces()
        self.draw_plot(fig)
        self.screen.blit(self.plot_surface, [0, 0])
        pygame.display.flip()
        self.check_exit()

    def check_exit(self):
        event_list = pygame.event.get()
        for event in event_list:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
