## Intro

This article describes popular game genres and research platforms, used in the literatures.

## Reference

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8632747



## Notes

### Arcade Games

Classic arcade games, of the type found in the late seventies' and eighties' arcade cabinets, home video game consoles and home computers.

- Atari 2600: Arcade Learning Environment (ALE)
  - ALE is built on top of the Atari 2600 emulator Stella and contains more than 50 original Atari 2600 games. The framework extracts the game score, 160×210 screen pixels and the RAM content that can be used as input for game playing agents.
- Nintendo NES: Retro Learning Environment (RLE)
  - currently contains seven games released for the Super Nintendo Entertainment System (SNES) . Many of these games have 3D graphics and the controller allows for over 720 action combinations. SNES games are thus more **complex** and **realistic** than Atari 2600 games but RLE has not been as popular as ALE.
- Commodore 64
- ZX Spectrum

In this category of games, it hugely relies on the graphic logics, such as reflecting the ball using a bar(Breakout). Most games require fast reactions and precise timing, and a few games, in particular, early sports games such as Track & Field (Konami, 1983) rely almost exclusively on speed and reactions. And another common requirement is navigating mazes or other complex 2D environments, e,g,. Pac-Man (Namco, 1980) and Boulder Dash (First Star Software, 1984). Among these games, some games have outstanding features, such as Montezuma’s Revenge (Parker Brothers, 1984), require long-term planning involving the memorization of temporarily unobservable game states.

### Racing Games

Racing games are games where the player is tasked with controlling some kind of vehicle or character so as to reach a goal in the shortest possible time, or as to traverse as far as possible along a track in a given time. And The vast majority of racing games take a continuous input signal as steering input, similar to a steering wheel. A popular environment for visual reinforcement learning with realistic 3D graphics is the open racing car simulator TORCS.

### First-Person Shooters (FPS)

FPSes have 3D graphics with partially observable states and are thus a more realistic environment to study. Usually, the viewpoint is that of the playercontrolled character, though some games that are broadly in the FPS categories adopt an over-the-shoulder viewpoint.

- ViZDoom, a framework that allows agents to play the classic first-person shooter Doom (id Software, 1993–2017)
- DeepMind Lab is a platform for 3D navigation and puzzlesolving tasks based on the Quake III Arena (id Software, 1999) engine 

### Open-World Games

- Minecraft (Mojang, 2011)
  - Project Malmo is a platform built on top of the open-world game Minecraft, which can be used to define many diverse and complex problems
- Grand Theft Auto (Rockstar Games, 1997–2013)

### Real-time Strategy Games

Strategy games are games where the player controls multiple characters or units, and the objective of the game is to prevail in some sort of conquest or conflict.

- StarCraft (Blizzard Entertainment, 1998–2017)
  - The Brood War API (BWAPI) enables software to communicate with StarCraft while the game runs, e.g. to extract state features and perform actions
  - DeepMind and Blizzard (the developers of StarCraft) have developed a machine learning API to support research in StarCraft II with features such as simplified visuals designed for convolutional networks

### Team Sports Games

Popular sports games are typically based on team-based sports such as soccer, basketball, and football. And they are meant to be a quite realistic.

- the annual Robot World Cup Soccer Games (RoboCup)
- Keepaway Soccer
- RoboCup 2D Half-Field-Offense (HFO)

### Text Adventure Games

