#!/usr/bin/env python
import os
os.environ["MUJOCO_GL"] = "glfw"  # GLFW 백엔드 사용

import sys
import glfw
import OpenGL.GL as gl
import numpy as np
import mujoco
from Env import OneLegEnv
from stable_baselines3 import PPO

def test_trained_model_manual_camera(model_path):
    env = OneLegEnv("xml/scene.xml")
    obs, _ = env.reset()

    # 학습된 모델 로드 (매개변수로 입력된 파일명을 사용)
    model = PPO.load(model_path)

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    window = glfw.create_window(1000, 800, "One Leg Reinforcement Learning", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    paused = False

    # 먼저 카메라 및 렌더 관련 객체 생성 (콜백 함수들에서 사용하기 위해 미리 선언)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(env.model, maxgeom=1000)
    context = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    # 카메라 기본 설정
    cam.azimuth = 90      # 초기 회전 각
    cam.elevation = -20   # 초기 각도
    cam.distance = 1.5
    cam.lookat = np.array([0.0, 0.0, 0.35])  # 초기 lookat 위치

    # ----- 마우스 인터랙션 관련 변수 -----
    left_mouse_pressed = False
    prev_left_mouse_x = None
    prev_left_mouse_y = None

    right_mouse_pressed = False
    prev_right_mouse_x = None
    prev_right_mouse_y = None

    # 마우스 버튼 콜백
    def mouse_button_callback(window, button, action, mods):
        nonlocal left_mouse_pressed, right_mouse_pressed
        nonlocal prev_left_mouse_x, prev_left_mouse_y
        nonlocal prev_right_mouse_x, prev_right_mouse_y
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                left_mouse_pressed = True
                prev_left_mouse_x, prev_left_mouse_y = glfw.get_cursor_pos(window)
            elif action == glfw.RELEASE:
                left_mouse_pressed = False
                prev_left_mouse_x, prev_left_mouse_y = None, None
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                right_mouse_pressed = True
                prev_right_mouse_x, prev_right_mouse_y = glfw.get_cursor_pos(window)
            elif action == glfw.RELEASE:
                right_mouse_pressed = False
                prev_right_mouse_x, prev_right_mouse_y = None, None
    glfw.set_mouse_button_callback(window, mouse_button_callback)

    # 커서 위치 콜백: 마우스 드래그에 따른 카메라 파라미터 업데이트
    def cursor_position_callback(window, xpos, ypos):
        nonlocal left_mouse_pressed, prev_left_mouse_x, prev_left_mouse_y
        nonlocal right_mouse_pressed, prev_right_mouse_x, prev_right_mouse_y, cam

        # 좌클릭 드래그: 회전 업데이트
        if left_mouse_pressed and prev_left_mouse_x is not None and prev_left_mouse_y is not None:
            dx = xpos - prev_left_mouse_x
            dy = ypos - prev_left_mouse_y
            cam.azimuth -= dx * 0.1
            cam.elevation -= dy * 0.1
            prev_left_mouse_x, prev_left_mouse_y = xpos, ypos

        # 우클릭 드래그: lookat 업데이트 (팬)
        if right_mouse_pressed and prev_right_mouse_x is not None and prev_right_mouse_y is not None:
            dx = xpos - prev_right_mouse_x  # 마우스 좌우 이동량
            dy = ypos - prev_right_mouse_y  # 마우스 상하 이동량

            # 좌우 panning: 현재 카메라 azimuth 기준 좌측 벡터 계산 (up = [0,0,1])
            horizontal_sensitivity = 0.003
            theta = np.radians(cam.azimuth)
            left_vector = np.array([-np.sin(theta), np.cos(theta), 0])
            cam.lookat[0] += left_vector[0] * dx * horizontal_sensitivity
            cam.lookat[1] += left_vector[1] * dx * horizontal_sensitivity

            # 상하 panning: lookat의 z (높이) 업데이트
            vertical_sensitivity = 0.003
            cam.lookat[2] += dy * vertical_sensitivity

            prev_right_mouse_x, prev_right_mouse_y = xpos, ypos

    glfw.set_cursor_pos_callback(window, cursor_position_callback)

    # 스크롤 콜백: 확대/축소 (zoom)
    def scroll_callback(window, xoffset, yoffset):
        nonlocal cam
        zoom_sensitivity = 0.1
        cam.distance -= yoffset * zoom_sensitivity
        if cam.distance < 0.1:
            cam.distance = 0.1
    glfw.set_scroll_callback(window, scroll_callback)

    # 키보드 콜백: 스페이스, F, C 키 처리
    def key_callback(window, key, scancode, action, mods):
        nonlocal paused, opt
        if action == glfw.PRESS:
            if key == glfw.KEY_SPACE:
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == glfw.KEY_F:
                current = opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]
                opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0 if current else 1
                print("Contact force visualization:", opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE])
            elif key == glfw.KEY_C:
                current = opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]
                opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0 if current else 1
                print("Contact point visualization:", opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT])
    glfw.set_key_callback(window, key_callback)

    print("학습된 모델 테스트 시작 - 창을 닫을 때까지 시뮬레이션이 실행됩니다.")
    while not glfw.window_should_close(window):
        glfw.poll_events()
        if not paused:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

        mujoco.mjv_updateScene(env.model, env.data, opt, None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        gl.glViewport(0, 0, viewport_width, viewport_height)
        gl.glClearColor(0.6, 0.6, 0.6, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, viewport_width, viewport_height),
                          scene, context)
        glfw.swap_buffers(window)

    glfw.terminate()
    print("학습된 모델 테스트를 종료합니다.")

if __name__ == "__main__":
    # 명령줄 인수를 통해 모델 파일명을 입력받습니다.
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "ppo_admittance"  # 기본 모델 이름
    model_path = os.path.join("models", model_name)
    test_trained_model_manual_camera(model_path)
