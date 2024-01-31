# OK Image Translator 👌
In response to the growing trend of drop shipping services importing cost-effective products from China, we have developed an automated translation system for product detail pages. This innovation significantly reduces the need for manual or semi-automatic translation efforts. Our system is designed to streamline the translation process, making international e-commerce more accessible and efficient. It bridges the language barrier in online shopping, facilitating smoother cross-border transactions.

# Technologies used 📟
Our project leverages cutting-edge technologies to deliver clean and precise translations of product details. Here's how it works:

- **Text Detection & Recognition**: Utilizing Paddle OCR, our system expertly identifies and extracts text from product detail pages. Paddle OCR's advanced algorithms ensure high accuracy in recognizing diverse fonts and layouts.
    
- **Advanced Image Processing**: After text detection, we employ lama inpainting to process the areas within the detected bounding boxes. This step ensures that the underlying images are seamlessly reconstructed, maintaining the visual integrity of the original page.

- **Translation with OpenAI's Language Model**: Following the image processing step, the extracted text is translated using OpenAI's advanced language models. These models provide highly accurate and contextually appropriate translations, ensuring that the essence and nuances of the original text are preserved. This step is crucial in maintaining the accuracy and reliability of the translated content, catering to a global audience with diverse linguistic needs.

- **Text Arrangement and Styling**: Post text extraction and image processing, we use clustering algorithms to meticulously organize the translated text. This includes aligning the text appropriately, adjusting sizes for consistency, and matching the original color schemes. The result is a translated page that not only conveys the right information but also maintains the aesthetic appeal of the original design.

By integrating these technologies, our system provides a polished and professional translation of product detail pages, significantly reducing the time and effort involved in manual translations.

# Environment setup ⚙️
**Clone the repo** : `git clone https://github.com/nexacorn/image-translator.git`

**Make `.env` file in `image-translator`folder**:

```
# .env

OPENAI_KEY="Your Open AI Key"
```

**Set Conda environment**

```
$ conda create -n translate python=3.8 -y
$ sh setting.sh
```

❗️ **Check `setting.sh`** : Select the appropriate code for your environment, either GPU or CPU based. If you plan to use a GPU, ensure to install the torch packages compatible with your CUDA version. **Currently, we are using CUDA 11.8.** Tailor your setup to match these specifications for optimal performance.

# Execution 
**Run the flask server**

```
$ python3 flask_translate.py
```

**POST Request**

```json
{
	"urls" : ["image url 1", "image url 2", ...],
	"language" : "vi or ko"
}
```

**Curl**

```
curl --request POST \ 
--url http://127.0.0.1:8088/process-images \ 
--header 'Content-Type: application/json' \ 
--header 'User-Agent: insomnia/8.6.0' \ 
--data '{ "urls":["image url 1", "image url 2"], "language": "vi or ko" }'
```

# Results 
**CHINESE  to VIETNAM**

<table width="100%">
<tr>
	<td width=50%>
		  <img src="https://okit.s3.ap-northeast-2.amazonaws.com/git_1.jpeg">
	</td>
	<td width=50%>
		  <img src="https://okit.s3.ap-northeast-2.amazonaws.com/git_1_result.png">
	</td>
</tr>
</table>
<table width="100%">
<tr>
	<td width=50%>
		  <img src="https://okit.s3.ap-northeast-2.amazonaws.com/image_3.jpg">
	</td>
	<td width=50%>
		  <img src="https://okit.s3.ap-northeast-2.amazonaws.com/git_3_result.png">
	</td>
</tr>
</table>

**CHINESE  to KOREA**

<table width="100%">
<tr>
	<td width=50%>
		  <img src="https://okit.s3.ap-northeast-2.amazonaws.com/git_1.jpeg">
	</td>
	<td width=50%>
		  <img src="https://okit.s3.ap-northeast-2.amazonaws.com/ko_git_1.png">
	</td>
</tr>
</table>
<table width="100%">
<tr>
	<td width=50%>
		  <img src="https://okit.s3.ap-northeast-2.amazonaws.com/image_3.jpg">
	</td>
	<td width=50%>
		  <img src="https://okit.s3.ap-northeast-2.amazonaws.com/ko_git_3.png">
	</td>
</tr>
</table>

**Resources** 
https://www.vvic.com/item/65116e755a2ea200083244c7
https://www.vvic.com/item/65659fe0c6dec70008460df0

# Contributing 🔥
We warmly welcome contributions to the OK Image Translator project. If you're interested in helping, here are some ways you can contribute:

- **Reporting Bugs**: If you find a bug, please open an issue with a clear description of the problem and a code sample or an executable test case demonstrating the expected behavior that is not occurring.
    
- **Feature Requests**: You can also open issues for feature requests. Please include as much detail as possible about the feature and why it would be useful.
    
- **Submitting Pull Requests**: We are always happy to receive pull requests. Please ensure that your code adheres to the project's coding standards and include tests for your changes.

# Acknowledgments 😁
- **OCR** : https://github.com/PaddlePaddle/PaddleOCR.git
- **LaMa** : https://github.com/advimman/lama.git
