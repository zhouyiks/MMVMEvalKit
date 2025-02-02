from vlmeval.smp import *
from vlmeval.api.base import BaseAPI

headers = 'Content-Type: application/json'


class GeminiWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'gemini-1.0-pro',
                 retry: int = 5,
                 wait: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 temperature: float = 0.0,
                 system_prompt: str = None,
                 max_tokens: int = 1024,
                 proxy: str = None,
                 backend='genai',
                 project_id='vlmeval',
                 **kwargs):

        assert model in ['gemini-1.0-pro', 'gemini-1.5-pro', 'gemini-1.5-flash']

        self.model = model
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        if key is None:
            key = os.environ.get('GOOGLE_API_KEY', None)
        # Try to load backend from environment variable
        be = os.environ.get('GOOGLE_API_BACKEND', None)
        if be is not None and be in ['genai', 'vertex', 'aigc369']:
            backend = be

        assert backend in ['genai', 'vertex', 'aigc369']
        if backend == 'genai':
            # We have not evaluated Gemini-1.5 w. GenAI backend
            assert key is not None  # Vertex does not require API Key

        self.backend = backend
        self.project_id = project_id
        self.api_key = key

        self.img_size = 512
        self.img_detail = 'low'

        if proxy is not None:
            proxy_set(proxy)
        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    def build_msgs_genai(self, inputs):
        messages = [] if self.system_prompt is None else [self.system_prompt]
        for inp in inputs:
            if inp['type'] == 'text':
                messages.append(inp['value'])
            elif inp['type'] == 'image':
                messages.append(Image.open(inp['value']))
        return messages

    def build_msgs_vertex(self, inputs):
        from vertexai.generative_models import Part, Image
        messages = [] if self.system_prompt is None else [self.system_prompt]
        for inp in inputs:
            if inp['type'] == 'text':
                messages.append(inp['value'])
            elif inp['type'] == 'image':
                messages.append(Part.from_image(Image.load_from_file(inp['value'])))
        return messages
    
    def build_msg_aigc369(self, inputs):
        messages = [] if self.system_prompt is None else [self.system_prompt]
        content = []
        for inp in inputs:
            if inp['type'] == 'text':
                content.append(dict(type='text', text=inp['value']))
            elif inp['type'] == 'image':
                from PIL import Image
                img = Image.open(inp['value'])
                b64 = encode_image_to_base64(img, target_size=self.img_size)
                img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail=self.img_detail)
                content.append(dict(type='image_url', image_url=img_struct))
        messages.append(dict(role='user', content=content))

    def generate_inner(self, inputs, **kwargs) -> str:
        if self.backend == 'genai':
            import google.generativeai as genai
            assert isinstance(inputs, list)
            pure_text = np.all([x['type'] == 'text' for x in inputs])
            genai.configure(api_key=self.api_key)

            if pure_text and self.model == 'gemini-1.0-pro':
                model = genai.GenerativeModel('gemini-1.0-pro')
            else:
                assert self.model in ['gemini-1.5-pro', 'gemini-1.5-flash']
                model = genai.GenerativeModel(self.model)

            messages = self.build_msgs_genai(inputs)
            gen_config = dict(max_output_tokens=self.max_tokens, temperature=self.temperature)
            gen_config.update(kwargs)
            try:
                answer = model.generate_content(
                    messages,
                    generation_config=genai.types.GenerationConfig(**gen_config)).text
                return 0, answer, 'Succeeded! '
            except Exception as err:
                if self.verbose:
                    self.logger.error(err)
                    self.logger.error(f'The input messages are {inputs}.')

                return -1, '', ''
        elif self.backend == 'vertex':
            import vertexai
            from vertexai.generative_models import GenerativeModel
            vertexai.init(project=self.project_id, location='us-central1')
            model_name = 'gemini-1.0-pro-vision' if self.model == 'gemini-1.0-pro' else self.model
            model = GenerativeModel(model_name=model_name)
            messages = self.build_msgs_vertex(inputs)
            print(messages)
            exit(0)
            try:
                resp = model.generate_content(messages)
                answer = resp.text
                return 0, answer, 'Succeeded! '
            except Exception as err:
                if self.verbose:
                    self.logger.error(err)
                    self.logger.error(f'The input messages are {inputs}.')

                return -1, '', ''
        elif self.backend == 'aigc369':
            import http.client
            import json
            from openai import OpenAI
            
            messages = self.build_msg_aigc369(inputs)

            client = OpenAI(
                base_url="https://api.aigc369.com/v1",
                api_key="sk-rG3TiRockhZQlhKn613670F4D7284a33B6De8c6e0d742eE7"
            )

            response = client.chat.completions.create(
                model="gemini-1.5-pro-latest",
                messages=messages,
                temperature=0.9,
                max_tokens=1024,
            )

            answer = response.choices[0]
            # print(answer)
            # exit(0)
            return 0, answer, 'Succeeded! '



class GeminiProVision(GeminiWrapper):



    def generate(self, message, dataset=None):
        return super(GeminiProVision, self).generate(message)
