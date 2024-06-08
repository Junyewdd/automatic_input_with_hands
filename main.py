from show_screen import show_let_start_screen, show_hi_screen
from gesture_use import recognize_gestures, use_gestures, input_shortcut

if __name__ == "__main__":
    # 프로그램 시작 화면
    choice = show_hi_screen()
    
    # 단축어 저장
    if choice == 1:
        recognize_gestures()
    elif choice == 2:
        input_shortcut()
        

    # # 필요없는 단축어 삭제
    # delete_sentences()
    
    # 사용 시작 화면
    show_let_start_screen()

    # 단축어 사용
    use_gestures()

    # # Step 4: Print all completed sentences
    # print_completed_sentences()