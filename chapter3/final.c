#include <stdio.h>

int main() {
    // 배열 a를 선언 및 초기화
    int a[] = {11, 22, 33, 44, 55, 66};
    
    // 배열 크기 계산
    int length = sizeof(a) / sizeof(a[0]);
    
    // 포인터 p는 배열의 첫 번째 원소 주소 저장
    int *p = a;           
    
    // 포인터 q는 배열의 마지막 원소 주소 저장
    int *q = a + length - 1;
    
    // 사용자 이름 출력 (필요 시 본인 이름으로 변경)
    printf("손정빈씨 출력 결과는 다음과 같습니다.!\n");
    
    // 포인터 q부터 p까지 역순으로 출력
    // q가 p보다 앞으로 올 때까지 반복
    while (q >= p) {
        printf("%d ", *q);  // q가 가리키는 값을 출력
        q--;                // q를 한 칸 앞으로 이동 (역방향)
    }
    
    printf("\n");
    return 0;
}