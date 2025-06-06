import cv2
import numpy as np
import time
import argparse
from datetime import datetime

def mosaic(img, rect, size=20):
    """지정된 영역에 모자이크 효과를 적용합니다"""
    # 모자이크 처리할 영역 추출
    (x, y, w, h) = rect
    # 좌표 범위 확인 및 조정
    x, y = max(0, x), max(0, y)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)
    
    # 관심 영역 추출
    roi = img[y:y+h, x:x+w]
    
    # 축소 후 확대하여 모자이크 효과 생성
    roi = cv2.resize(roi, (w // size, h // size))
    roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 원본 이미지에 모자이크 적용된 부분 복사
    img[y:y+h, x:x+w] = roi
    
    return img

def blur_face(img, rect, blur_factor=15):
    """가우시안 블러 효과를 적용합니다"""
    (x, y, w, h) = rect
    # 좌표 범위 확인 및 조정
    x, y = max(0, x), max(0, y)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)
    
    # 관심 영역 추출
    roi = img[y:y+h, x:x+w]
    
    # blur_factor가 홀수가 아니면 홀수로 조정
    if blur_factor % 2 == 0:
        blur_factor += 1
    
    # 가우시안 블러 적용
    roi = cv2.GaussianBlur(roi, (blur_factor, blur_factor), 0)
    
    # 원본 이미지에 블러 적용된 부분 복사
    img[y:y+h, x:x+w] = roi
    
    return img

def parse_args():
    parser = argparse.ArgumentParser(description='실시간 얼굴 감지 및 모자이크/블러 처리 시스템')
    parser.add_argument('--camera', type=int, default=0, help='카메라 장치 번호 (기본값: 0)')
    parser.add_argument('--mode', type=str, default='mosaic', choices=['mosaic', 'blur'], help='효과 모드 (mosaic 또는 blur)')
    parser.add_argument('--strength', type=int, default=20, help='모자이크/블러 강도 (기본값: 20)')
    parser.add_argument('--min-size', type=int, default=20, help='감지할 최소 얼굴 크기 (기본값: 20)')
    parser.add_argument('--scale', type=float, default=1.1, help='이미지 스케일 감소 비율 (기본값: 1.1)')
    parser.add_argument('--neighbors', type=int, default=4, help='minNeighbors 파라미터 (기본값: 4)')
    parser.add_argument('--save-video', action='store_true', help='비디오 저장 활성화')
    parser.add_argument('--debug', action='store_true', help='디버그 창 표시')
    return parser.parse_args()

def main():
    # 명령줄 인자 파싱
    args = parse_args()
    
    # 얼굴 감지기 로드 - Haar Cascade 사용
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 얼굴 감지기가 제대로 로드되었는지 확인
    if face_cascade.empty():
        print("얼굴 감지 분류기를 로드할 수 없습니다. OpenCV가 올바르게 설치되었는지 확인하세요.")
        return
    
    # 카메라 열기
    cap = cv2.VideoCapture(args.camera)
    
    # 카메라가 정상적으로 열렸는지 확인
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
    # 카메라 프레임 크기 및 FPS 가져오기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 비디오 저장 설정
    out = None
    if args.save_video:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f'face_detection_{args.mode}_{timestamp}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
        print(f"비디오를 {video_filename}에 저장합니다.")
    
    print(f"얼굴 감지 및 {args.mode} 처리 시작...")
    print("키 설명:")
    print("  'q' - 종료")
    print("  'm' - 모자이크 모드로 전환")
    print("  'b' - 블러 모드로 전환")
    print("  '+' - 효과 강도 증가")
    print("  '-' - 효과 강도 감소")
    print("  's' - 현재 프레임 저장")
    print("  'r' - 비디오 녹화 시작/중지")
    print("  'd' - 디버그 모드 토글")
    
    # 초기 설정
    effect_mode = args.mode
    effect_strength = args.strength
    # blur 모드일 경우 홀수값으로 조정
    if effect_mode == 'blur' and effect_strength % 2 == 0:
        effect_strength += 1
    is_recording = args.save_video
    show_debug = args.debug
    
    # 성능 측정 변수
    frame_count = 0
    start_time = time.time()
    face_detected = False
    
    # 비디오 루프
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
        
        # 처리 시작 시간
        process_start = time.time()
        
        # 그레이스케일 변환 (Haar Cascade용)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Haar Cascade로 얼굴 감지
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=args.scale,
            minNeighbors=args.neighbors,
            minSize=(args.min_size, args.min_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # 원본 프레임 복사 (UI 및 마커에 사용)
        display_frame = frame.copy() if show_debug else None
        
        # 감지된 모든 얼굴에 효과 적용
        face_count = 0
        for (x, y, w, h) in faces:
            # 얼굴 영역을 약간 확장
            expand = int(w * 0.1)  # 10% 확장
            x_expanded = max(0, x - expand)
            y_expanded = max(0, y - expand)
            w_expanded = w + expand * 2
            h_expanded = h + expand * 2
            
            # 효과 적용
            if effect_mode == 'mosaic':
                mosaic(frame, (x_expanded, y_expanded, w_expanded, h_expanded), size=effect_strength)
            else:  # 'blur'
                blur_face(frame, (x_expanded, y_expanded, w_expanded, h_expanded), blur_factor=effect_strength)
            
            # 디버그 모드에서 얼굴 영역 표시
            if show_debug and display_frame is not None:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            face_count += 1
            face_detected = True
        
        # 처리 시간 계산
        process_time = (time.time() - process_start) * 1000  # 밀리초로 변환
        
        # FPS 계산
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps_actual = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # 정보 텍스트 추가
        info_color = (0, 255, 0) if face_detected else (0, 0, 255)
        cv2.putText(frame, f'Faces: {face_count}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, info_color, 2)
        cv2.putText(frame, f'Mode: {effect_mode.capitalize()} ({effect_strength})', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f'FPS: {fps_actual:.1f}', (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f'Process: {process_time:.1f}ms', (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 녹화 중 표시
        if is_recording and out is not None:
            cv2.putText(frame, "REC", (frame_width - 70, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.circle(frame, (frame_width - 30, 25), 10, (0, 0, 255), -1)
            out.write(frame)
        
        # 결과 표시
        cv2.imshow('Face Detection & Privacy Protection', frame)
        
        # 디버그 창 표시
        if show_debug and display_frame is not None:
            cv2.imshow('Face Detection (Debug)', display_frame)
        
        # 키 입력 처리
        key = cv2.waitKey(1)
        if key == -1:  # 키 입력 없음
            continue
            
        # 키 코드 변환 - ASCII 코드만 추출
        key = key & 0xFF
        
        if key == ord('q'):  # 종료
            break
        elif key == ord('m'):  # 모자이크 모드로 전환
            effect_mode = 'mosaic'
            print(f"모드 변경: {effect_mode}")
        elif key == ord('b'):  # 블러 모드로 전환
            effect_mode = 'blur'
            # 블러 모드로 전환 시 홀수 강도로 조정
            if effect_strength % 2 == 0:
                effect_strength += 1
            print(f"모드 변경: {effect_mode}")
        elif key == ord('+') or key == ord('='):  # 효과 강도 증가
            if effect_mode == 'blur':
                # 블러 모드일 때는 홀수로 유지
                effect_strength = min(49, effect_strength + 2)
                if effect_strength % 2 == 0:
                    effect_strength += 1
            else:
                effect_strength = min(50, effect_strength + 2)
            print(f"효과 강도: {effect_strength}")
        elif key == ord('-') or key == ord('_'):  # 효과 강도 감소
            if effect_mode == 'blur':
                # 블러 모드일 때는 홀수로 유지
                effect_strength = max(3, effect_strength - 2)
                if effect_strength % 2 == 0:
                    effect_strength -= 1
            else:
                effect_strength = max(2, effect_strength - 2)
            print(f"효과 강도: {effect_strength}")
        elif key == ord('s'):  # 현재 프레임 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_filename = f'face_detection_{timestamp}.jpg'
            cv2.imwrite(img_filename, frame)
            print(f"이미지를 {img_filename}에 저장했습니다.")
        elif key == ord('r'):  # 녹화 시작/중지
            if out is None and not is_recording:
                # 녹화기 생성
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_filename = f'face_detection_{effect_mode}_{timestamp}.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
                is_recording = True
                print(f"녹화 시작: {video_filename}")
            elif out is not None and is_recording:
                # 녹화 중지
                out.release()
                out = None
                is_recording = False
                print("녹화 중지")
        elif key == ord('d'):  # 디버그 모드 토글
            show_debug = not show_debug
            if show_debug:
                print("디버그 모드 활성화")
            else:
                print("디버그 모드 비활성화")
                if cv2.getWindowProperty('Face Detection (Debug)', cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow('Face Detection (Debug)')
            
    # 자원 해제
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()