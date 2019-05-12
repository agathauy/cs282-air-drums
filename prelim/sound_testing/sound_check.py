import pygame
import time
#         self.directory_sound = "../../sounds/"

pygame.mixer.pre_init()
pygame.init()
pygame.mixer.init(frequency=22050,size=-16,channels=2,buffer=4096)
snare = pygame.mixer.Sound('../../sounds/snare.wav')
hihat = pygame.mixer.Sound('../../sounds/hihat.wav')
for i in range(5):
    snare.play()
    time.sleep(0.5)
    hihat.play()
    time.sleep(0.5)

snare.stop()
