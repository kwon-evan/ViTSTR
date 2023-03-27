from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    kor: bool = False  # use Korean character or not
    rgb: bool = False  # use rgb input
    pad: bool = False  # whether to keep ratio then pad for image resize
    pretrained: bool = False  # load pretrained model
    num_fiducial: int = 20  # number of fiducial points of TPS-STN
    batch_max_length: int = 10  # maximum label length
    img_size: Tuple[int, int] = (224, 224)  # image size (H, W)
    model_name: str = "vitstr_base_patch16_224"

    @property
    def character(self):
        if self.kor:
            return [
                # NUMBERS
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                # 자가용
                "가",
                "나",
                "다",
                "라",
                "마",
                "거",
                "너",
                "더",
                "러",
                "머",
                "버",
                "서",
                "어",
                "저",
                "고",
                "노",
                "도",
                "로",
                "모",
                "보",
                "소",
                "오",
                "조",
                "구",
                "누",
                "두",
                "루",
                "무",
                "부",
                "수",
                "우",
                "주",
                "기",
                "니",
                "디",
                "리",
                "미",
                "비",
                "시",
                "이",
                "지",
                # 영업용
                "바",
                "사",
                "아",
                "자",
                "카",
                "파",
                "타",
                "차",
                # 영업용(택배)
                "배",
                # 영업용(건설)
                "영",
                # 렌터카
                "하",
                "허",
                "호",
                "히",
                # 육군, 공군, 해군
                "육",
                "공",
                "해",
                # 국방부 및 직할부대 등
                "국",
                "합",
                # 도시
                "강원",
                "경기",
                "경남",
                "경북",
                "광주",
                "대구",
                "대전",
                "부산",
                "서울",
                "인천",
                "전남",
                "전북",
                "제주",
                "충남",
                "충북",
                # 도시 sub
                "강남",
                "강서",
                "계양",
                "고양",
                "관악",
                "광명",
                "구로",
                "금천",
                "김포",
                "남동",
                "동대문",
                "동작",
                "미추홀",
                "부천",
                "부평",
                "서대문",
                "서초",
                "안산",
                "안양",
                "양천",
                "연수",
                "영등포",
                "용산",
                "인천",
                "중",
            ]
        else:
            return "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # character label
