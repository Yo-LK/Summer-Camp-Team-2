import json
from typing import Optional, Dict, List, Any


# ==========================================
# 1. Agent Skill 模块定义 (Restaurant Selection Tool Skill)
#    1. Agent Skill Module Definition (Restaurant Selection Tool Skill)
#    1. Agent Skill 모듈 정의 (식당 선택 Tool Skill)
# ==========================================

def restaurant_recommender_skill(
        cuisine: str,
        tier: str,
        data_path: str = "restaurants_100yuan.1.json"
) -> Optional[Dict[str, Any]]:
    """
    Agent Skill: 餐厅选择与推荐技能 (Tool Skill)
    Agent Skill: Restaurant selection and recommendation skill (Tool Skill)
    Agent Skill: 식당 선택 및 추천 스킬 (Tool Skill)

    【原理/能力】
    【Principle/Capability】
    【원리/기능】
    接收结构化的菜系与档次需求，对 JSON 数据库进行检索过滤：
    Receives structured cuisine and tier requirements, searches and filters the JSON database:
    구조화된 요리 종류 및 등급 요구사항을 받아 JSON 데이터베이스를 검색 및 필터링:
    - 若选择"低档"，在同菜系中选择平均价格（average_price）最低的餐厅；
    - If "low" tier is selected, picks the restaurant with the lowest average_price in the same cuisine;
    - "저급(low)" 선택 시, 동일 요리 종류 내에서 평균 가격(average_price)이 가장 낮은 식당 선택;
    - 若选择"高档"，在同菜系中选择平均价格（average_price）最高的餐厅。
    - If "high" tier is selected, picks the restaurant with the highest average_price in the same cuisine.
    - "고급(high)" 선택 시, 동일 요리 종류 내에서 평균 가격(average_price)이 가장 높은 식당 선택.

    :param cuisine: 目标菜系英文名 / Target cuisine name in English / 목표 요리 종류 영문명 ("Chinese", "Korean", "Singaporean")
    :param tier: 餐厅档次 / Restaurant tier / 식당 등급 ("high" or "low")
    :param data_path: JSON 数据文件路径 / JSON data file path / JSON 데이터 파일 경로
    :return: 匹配到的餐厅完整字典数据（包含 restaurant_id），若无符合条件的餐厅则返回 None
             Full dictionary data of matched restaurant (including restaurant_id), returns None if no match
             매칭된 식당의 전체 딕셔너리 데이터 (restaurant_id 포함), 조건에 맞는 식당이 없으면 None 반환
    """
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            restaurants: List[Dict[str, Any]] = json.load(f)
    except FileNotFoundError:
        print(f"[Agent Skill Exec Error] File not found / 找不到数据文件 / 데이터 파일을 찾을 수 없습니다: {data_path}")
        return None

    # 按菜系进行过滤 / Filter by cuisine / 요리 종류별 필터링
    filtered = [r for r in restaurants if r.get("cuisine", "").lower() == cuisine.lower()]

    if not filtered:
        return None

    # 根据档次选择价格最高或最低的餐厅
    # Select the restaurant with highest or lowest price based on tier
    # 등급에 따라 가격이 가장 높거나 낮은 식당 선택
    if tier == "high":
        best_match = max(filtered, key=lambda x: x.get("average_price", 0))
    else:  # tier == "low"
        best_match = min(filtered, key=lambda x: x.get("average_price", float("inf")))

    return best_match


# ==========================================
# 2. 交互与输入校验模块 (Input Handling)
#    2. Interactive Input Handling & Validation
#    2. 상호작용 및 입력 검증 모듈
# ==========================================

def get_valid_user_input() -> tuple[str, str]:
    """
    循环提示用户输入，严格校验以下错误输入情况：
    Loop to prompt user for input, strictly validating input errors:
    사용자 입력을 반복 요청하며 다음과 같은 오류 입력을 엄격히 검증:
    1. 输入参数缺失或过多（少输入 / 多输入） -> 提醒并要求重新输入
       Missing or too many parameters -> Alert and prompt to re-enter
       파라미터 누락 또는 초과 -> 경고 후 재입력 요청
    2. 输入了不支持的菜系 -> 提醒并要求重新输入
       Unsupported cuisine entered -> Alert and prompt to re-enter
       지원되지 않는 요리 종류 입력 -> 경고 후 재입력 요청
    3. 档次输入错误 -> 提醒并要求重新输入
       Invalid tier entered -> Alert and prompt to re-enter
       등급 입력 오류 -> 경고 후 재입력 요청
    """
    cuisine_map = {
        "chinese": "Chinese",
        "korean": "Korean",
        "singaporean": "Singaporean"
    }

    tier_map = {
        "high": "high",
        "low": "low"
    }

    while True:
        print("\n--------------------------------------------------")
        # 按英文提示输入 / Prompt input in English / 영문 입력 안내
        print("Please enter the cuisine you want to eat (Chinese, Korean, Singaporean) and the tier (High, Low).")
        print("Hint: Please separate the two parameters with a space, e.g., 'Chinese High'")

        user_input = input("User Input: ").strip()
        parts = user_input.split()

        # 校验 1：输入参数数量检查（必须正好 2 个）
        # Validation 1: Check parameter count (must be exactly 2)
        # 검증 1: 입력 파라미터 개수 확인 (반드시 2개여야 함)
        if len(parts) != 2:
            print("[Input Error] Please enter exactly 2 items (cuisine and high/low tier), separated by a space!")
            continue

        raw_cuisine, raw_tier = parts[0].lower(), parts[1].lower()

        # 校验 2：菜系合法性检查
        # Validation 2: Cuisine validity check
        # 검증 2: 요리 종류 유효성 확인
        if raw_cuisine not in cuisine_map:
            print(f"[Input Error] Cuisine '{parts[0]}' is not supported. Please try again (Supported: Chinese, Korean, Singaporean)!")
            continue

        # 校验 3：高低档合法性检查
        # Validation 3: Tier validity check
        # 검증 3: 등급 유효성 확인
        if raw_tier not in tier_map:
            print("[Input Error] Tier must be either 'High' or 'Low'!")
            continue

        # 转换为规范参数 / Map to standardized parameters / 표준 파라미터로 변환
        mapped_cuisine = cuisine_map[raw_cuisine]
        mapped_tier = tier_map[raw_tier]

        return mapped_cuisine, mapped_tier


# ==========================================
# 3. 主流程与 Agent Skill Demo 展示
#    3. Main Process & Agent Skill Demo
#    3. 메인 프로세스 및 Agent Skill Demo 출력
# ==========================================

def main() -> str:
    print("=== Welcome to the Agent Restaurant Recommendation System Demo ===")

    # 1. 交互并获取合规的用户输入
    # 1. Interact and obtain valid user input
    # 1. 상호작용을 통해 유효한 사용자 입력 획득
    cuisine, tier = get_valid_user_input()

    # 2. 显示 Agent Skill 的调用 Demo
    # 2. Display Agent Skill execution demo
    # 2. Agent Skill 호출 Demo 표시
    print(f"\n[Agent Skill Trigger Demo]")
    print(f"  └─ Triggering Agent Tool: restaurant_recommender_skill")
    print(f"  └─ Arguments passed: cuisine='{cuisine}', tier='{tier}'")

    # 3. 执行 Agent Skill
    # 3. Execute Agent Skill
    # 3. Agent Skill 실행
    selected_restaurant = restaurant_recommender_skill(cuisine, tier, data_path="restaurants_100yuan.1.json")

    if selected_restaurant:
        restaurant_name = selected_restaurant.get("name", "Unknown")
        restaurant_id = selected_restaurant.get("restaurant_id", "")

        # 4. 按要求格式化输出
        # 4. Format output as required
        # 4. 요구사항에 맞게 출력 형식화
        print(f"\nOutput: Recommended restaurant is {restaurant_name}.")
        print(f"[Agent Skill Return Demo] Returned restaurant ID: {restaurant_id}")

        # 返回选择的餐厅 id（供另一个 py 文件调用）
        # Return selected restaurant id (for invocation by another py file)
        # 선택된 식당 ID 반환 (다른 py 파일에서 호출용)
        return restaurant_id
    else:
        print("\nOutput: No restaurant found matching the criteria.")
        return ""


if __name__ == "__main__":
    # 运行演示并返回 id
    # Run demo and return id
    # 데모 실행 및 ID 반환
    returned_id = main()
    print(f"\n[For External Call] Returned ID: {returned_id}")
