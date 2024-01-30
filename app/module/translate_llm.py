from openai import OpenAI
import re


def contains_english(s):
    return bool(re.fullmatch("[a-zA-Z0-9\s!@#$%^&*()_+\-=[\]{ };'\":,.<>/?\\|￥]+", s))


def translate_llm(original_text_len, orginal_text, openai_api_key, lang="ko"):
    client = OpenAI(api_key=openai_api_key)
    if "ko" in lang:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "배열로 들어온 총 "
                    + str(original_text_len)
                    + "개의 유저 메시지는 패션상품의 광고문구야 무조건 빠짐없이 모든 메시지들을 번역해야돼. 즉, "
                    + str(original_text_len)
                    + '개의 번역결과가 나와야해. 번역이 잘 안되는 것들은 그대로 순서에 맞게 반환해줘. 이것들을 패션 상품에 어울리는 한글로, 가능한 개조식으로 번역해줘. 배열 형식과 순서는 유지한채 json 형식으로, 한글로 번역해서 반환해줘. return example : { translated : ["번역 결과","번역 결과","번역 결과","번역 결과"] }',
                },
                {"role": "user", "content": orginal_text},
            ],
        )
        return completion
    elif "vi" in lang:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Trong tổng số "
                    + str(original_text_len)
                    + " tin nhắn người dùng nhập từ mảng là slogan quảng cáo cho sản phẩm thời trang, cần phải dịch tất cả mà không bỏ sót bất kỳ tin nhắn nào. Nghĩa là, phải có "
                    + str(original_text_len)
                    + ' kết quả dịch. Những tin nhắn khó dịch thì trả lại nguyên trạng theo đúng thứ tự. Hãy dịch những điều này sang tiếng Việt phù hợp với sản phẩm thời trang, càng sáng tạo càng tốt. Giữ nguyên định dạng và thứ tự của mảng dưới dạng json, trả lại bằng tiếng Việt. Ví dụ trả về: { translated : ["kết quả dịch", "kết quả dịch", "kết quả dịch", "kết quả dịch"] }',
                },
                {"role": "user", "content": orginal_text},
            ],
        )
        return completion
