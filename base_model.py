import streamlit as st
import PIL
from funcs import load_css , linegaro , linesero , load_font , apply_custom_font
import webbrowser  
from css import hard_type_css

def base_model():
    a,b,c,d = st.columns([0.05, 0.1, 0.01 ,1])
    with a:
        st.write('')
        st.image('https://kli.korean.go.kr/images/logo_kor_2.png')
    with b:
        st.write('')
        st.write('')
        st.image('https://kli.korean.go.kr/images/logo_kor_header.png', width=400)
    with d:
        st.title('Base Model')
    st.markdown('#### 정의')
    st.markdown(''' 
                확실한 모니터링을 위해 앞서 이야기 한 전처리 방법을 적용시키지 않고 모델을 구축하였습니다. 
                또한 기본 모델의 데이터셋은 증강 없이 **nikluge-sa-2022-train** 데이터셋으로만 구성되어있으며, **nikluge-sa-2022-dev** 데이터셋을 이용하여 검증된 값을 기반으로 성능을 평가하였으며 학습이 완료 된 후, **nikluge-sa-2022-test** 에 적용시켜 제출한 결과입니다.
                ''')

    st.markdown('#')
    tab1 ,tab2, tab3 ,tab4= st.tabs(['데이터 전처리 및 로드', '모델 구축', '학습 및 검증','제출 결과'])
    with tab1:  
        st.markdown('''
                    ### 데이터 전처리 및 로드
                    ''')
        st.write('''
        ABSADataset 이라는 사용자 정의 데이터셋 클래스를 정의하여, ABSA(Aspect-Based Sentiment Analysis)를 위한 데이터 전처리와 토크나이징을 수행합니다. 
        이 클래스는 PyTorch의 Dataset 클래스를 상속하여, 모델 학습 시 배치 처리를 위한 인터페이스를 제공합니다.
        ''')
        st.markdown('''
                        #### 데이터셋 클래스 초기화 (`__init__`)
                        **주요 역할**: 데이터 파일을 읽고, JSON 형식의 데이터를 파싱하여 문장과 어노테이션을 처리합니다. 토크나이저를 사용하여 문장을 토큰화하고, 감성과 카테고리 레이블을 숫자로 변환합니다.
                        - 입력 매개변수:
                        - `file_path`: JSON 데이터 파일 경로.
                        - `tokenizer`: 텍스트를 토큰화하기 위한 토크나이저.
                        - `max_length`: 입력 텍스트의 최대 길이.
                        - `is_test`: 테스트 데이터 여부를 결정하는 플래그. True일 경우 어노테이션을 처리하지 않습니다.
        ''')
        st.code('''
        class ABSADataset(Dataset):
            def __init__(self, file_path, tokenizer, max_length=128, is_test=False):
                self.data = []  # 데이터를 저장할 리스트를 초기화합니다.
                self.tokenizer = tokenizer  # 텍스트를 토큰화하기 위한 토크나이저를 저장합니다.
                self.max_length = max_length  # 입력 시퀀스의 최대 길이를 설정합니다.
                self.is_test = is_test  # 테스트 데이터셋인지 여부를 나타내는 플래그를 설정합니다.

                # 감성 및 카테고리 레이블을 숫자로 매핑하는 사전 정의
                self.sentiment_map = {'positive': 0, 'neutral': 1, 'negative': 2}
                self.category_map = {
                    '제품 전체#품질': 0, '패키지/구성품#디자인': 1, '본품#일반': 2, '제품 전체#편의성': 3, '본품#다양성': 4,
                    '제품 전체#디자인': 5, '패키지/구성품#가격': 6, '본품#품질': 7, '브랜드#인지도': 8, '제품 전체#일반': 9,
                    '브랜드#일반': 10, '패키지/구성품#다양성': 11, '패키지/구성품#일반': 12, '본품#인지도': 13, '제품 전체#가격': 14,
                    '본품#편의성': 15, '패키지/구성품#편의성': 16, '본품#디자인': 17, '브랜드#디자인': 18, '본품#가격': 19,
                    '브랜드#품질': 20, '제품 전체#인지도': 21, '패키지/구성품#품질': 22, '제품 전체#다양성': 23, '브랜드#가격': 24
                }

                # JSONL 파일을 읽어 데이터를 파싱합니다.
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line.strip())  # 각 줄을 JSON 객체로 파싱합니다.
                        sentence = item['sentence_form']  # 문장을 추출합니다.
                        if not self.is_test:  # 테스트 데이터가 아닌 경우 어노테이션을 처리합니다.
                            for annotation in item['annotation']:
                                category, _, sentiment = annotation
                                if category in self.category_map:
                                    self.data.append({
                                        'sentence': sentence,
                                        'category': self.category_map[category],
                                        'sentiment': self.sentiment_map[sentiment]
                                    })
                        else:
                            # 테스트 데이터의 경우 어노테이션 없이 문장만 저장합니다.
                            self.data.append({'sentence': sentence})
                ''')

        st.markdown('''
                    #### 데이터셋 길이 반환 (`__len__`)
        **주요 역할**: 데이터셋의 총 샘플 수를 반환합니다. 모델 학습이나 평가 시, 데이터의 반복(iteration)을 설정하는 데 사용됩니다.
        ''')

        st.code('''
        def __len__(self):
            return len(self.data)
        ''')

        st.markdown('''
                    #### 특정 인덱스의 데이터 반환 (`__getitem__`)
        **주요 역할**: 주어진 인덱스에 해당하는 데이터 샘플을 반환합니다. 이 과정에서 텍스트를 토크나이징하여 모델 입력에 적합한 형태로 변환합니다.
        - 입력 텍스트를 토크나이저를 사용해 토큰화하고, `input_ids`, `attention_mask`를 생성합니다.
        - 테스트 데이터가 아닌 경우, 카테고리 및 감성 레이블을 반환합니다.
        ''')
        
        st.code('''
        def __getitem__(self, idx):
            item = self.data[idx]
            encoding = self.tokenizer(
                item['sentence'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()

            if not self.is_test:
                category = item['category']
                sentiment = item['sentiment']
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'category': torch.tensor(category, dtype=torch.long),
                    'sentiment': torch.tensor(sentiment, dtype=torch.long)
                }
            else:
                # 테스트 데이터일 경우 레이블 없이 입력만 반환합니다.
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }
        ''')

        st.markdown('''
        ### 감성과 카테고리 매핑 사전 (`sentiment_map` 및 `category_map`)
        **주요 역할**: 감성 및 카테고리 레이블을 정수형 값으로 매핑하는 사전을 정의하여, 데이터셋에서 문장의 어노테이션을 숫자 레이블로 변환합니다. 이는 모델의 출력과 손실 계산에 필요한 형태로 데이터를 변환하기 위함입니다.
        ''')

        st.code('''
        self.sentiment_map = {'positive': 0, 'neutral': 1, 'negative': 2}
        self.category_map = {
            '제품 전체#품질': 0, '패키지/구성품#디자인': 1, '본품#일반': 2, '제품 전체#편의성': 3, '본품#다양성': 4,
            '제품 전체#디자인': 5, '패키지/구성품#가격': 6, '본품#품질': 7, '브랜드#인지도': 8, '제품 전체#일반': 9,
            '브랜드#일반': 10, '패키지/구성품#다양성': 11, '패키지/구성품#일반': 12, '본품#인지도': 13, '제품 전체#가격': 14,
            '본품#편의성': 15, '패키지/구성품#편의성': 16, '본품#디자인': 17, '브랜드#디자인': 18, '본품#가격': 19,
            '브랜드#품질': 20, '제품 전체#인지도': 21, '패키지/구성품#품질': 22, '제품 전체#다양성': 23, '브랜드#가격': 24
        }
        ''')
        st.markdown('''
        ### 요약
        - `ABSADataset` 클래스는 **데이터 전처리, 토큰화, 레이블 변환**의 역할을 수행합니다.
        - 데이터를 JSON 형식으로 읽어 들인 후, 각 문장과 어노테이션을 파싱하여 모델 학습에 적합한 형태로 변환합니다.
        - 이를 통해 모델이 입력 텍스트와 그에 해당하는 레이블(감성과 카테고리)을 학습할 수 있도록 지원합니다.
    
        ''')
    with tab2:
        st.markdown('''
        #### 모델 구축
        ''')
        st.write('''
        ABSA(Aspect-Based Sentiment Analysis) 작업을 위한 모델을 정의합니다. 두 가지 모델인 `AspectCategoryModel`과 `SentimentModel`을 정의하고 각 모델은 특정한 예측 작업을 수행하도록 설계되어 있습니다.
        ''')
        st.markdown('''
        #### 흐름
        - **카테고리 예측 모델**: `AspectCategoryModel`은 주어진 문장에서 특정한 **카테고리**를 예측합니다.
        - **감성 예측 모델**: `SentimentModel`은 주어진 문장에서 특정한 **감성(긍정, 중립, 부정)** 을 예측합니다.
        - **모델 구조**: 두 모델 모두 **XLM-RoBERTa**와 같은 사전 학습된 트랜스포머 모델을 기반으로 합니다. 이 모델들은 텍스트 입력을 받아 `CLS` 토큰의 출력 벡터를 사용하여 각각의 예측을 수행합니다.
        ''')
        st.markdown('''
        #### `AspectCategoryModel` 정의
        **주요 역할**: 입력된 문장에 대해 **카테고리**를 예측하는 모델입니다.
        - **모델 초기화**: 사전 학습된 트랜스포머 모델(`XLM-RoBERTa`)을 로드하고, 최종 예측을 위한 선형 레이어(`Linear`)를 정의합니다.
        - **forward 메서드**: 텍스트 입력(`input_ids`)과 어텐션 마스크(`attention_mask`)를 받아 트랜스포머 모델을 통과시킨 후, `CLS` 토큰의 출력 벡터를 선형 레이어에 통과시켜 카테고리 예측을 수행합니다.
        ''')
        st.code('''
        import torch
        import torch.nn as nn
        from transformers import XLMRobertaModel

        class AspectCategoryModel(nn.Module):
            def __init__(self, model_name, num_labels):
                super(AspectCategoryModel, self).__init__()
                # 사전 학습된 XLM-RoBERTa 모델을 로드합니다.
                self.model = XLMRobertaModel.from_pretrained(model_name)
                # 카테고리 예측을 위한 선형 레이어를 정의합니다.
                self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
    
            def forward(self, input_ids, attention_mask):
                # XLM-RoBERTa 모델에 입력을 전달하여 출력값을 얻습니다.
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # 출력 중 CLS 토큰의 벡터를 가져옵니다.
                cls_output = outputs.last_hidden_state[:, 0, :]
                # CLS 토큰의 벡터를 선형 레이어에 전달하여 카테고리 예측값을 계산합니다.
                logits = self.classifier(cls_output)
                return logits
        ''')    
        st.markdown('''
            - `XLMRobertaModel`은 입력 텍스트를 임베딩하고 컨텍스트 정보를 반영한 벡터를 반환합니다.
            - `cls_output`은 `CLS` 토큰의 벡터로, 문장의 전체 의미를 요약한 정보를 담고 있습니다.
            - 최종적으로 `logits`는 각 카테고리에 대한 예측 확률을 나타내며, 모델의 출력으로 사용됩니다.
        ''')
        st.markdown('''
            ##### `SentimentModel` 정의
            **주요 역할**: 입력된 문장에 대해 **감성(긍정, 중립, 부정)** 을 예측하는 모델입니다.
            - `AspectCategoryModel`과 매우 유사한 구조로, 감성 예측을 위한 선형 레이어를 사용합니다.
            - 입력 텍스트의 `CLS` 토큰 벡터를 사용해 감성 레이블에 해당하는 예측값을 생성합니다.
        ''')
        st.code('''
        class SentimentModel(nn.Module):
            def __init__(self, model_name, num_labels):
                super(SentimentModel, self).__init__()
                
                # 사전 학습된 XLM-RoBERTa 모델을 로드합니다.
                self.model = XLMRobertaModel.from_pretrained(model_name)
                # 감성 예측을 위한 선형 레이어를 정의합니다.
                self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
    
            def forward(self, input_ids, attention_mask):
                # XLM-RoBERTa 모델에 입력을 전달하여 출력값을 얻습니다.
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # 출력 중 CLS 토큰의 벡터를 가져옵니다.
                cls_output = outputs.last_hidden_state[:, 0, :]
                # CLS 토큰의 벡터를 선형 레이어에 전달하여 감성 예측값을 계산합니다.
                logits = self.classifier(cls_output)
                return logits
        ''')    
        st.markdown('''
        - `SentimentModel`은 `AspectCategoryModel`과 동일한 방식으로 `CLS` 토큰의 벡터를 사용하지만, 감성 예측 레이블 수(`num_labels`)가 다릅니다.
        - `logits`는 각 감성(긍정, 중립, 부정)에 대한 예측 확률을 나타냅니다.
        ''')
        st.markdown('''
                    ### 모델의 공통 구조 설명
                    - **트랜스포머 모델**: 두 모델 모두 `XLMRobertaModel`을 사용하여, 사전 학습된 다국어 텍스트 처리 능력을 활용합니다. 이는 한국어 문장의 의미를 효과적으로 인코딩할 수 있게 해줍니다.
                    - **`CLS` 토큰 벡터 사용**: 트랜스포머 모델의 출력 중 첫 번째 토큰(`CLS`)의 벡터는 문장의 전체 의미를 표현하며, 이를 기반으로 예측을 수행합니다.
                    - **선형 레이어(`Linear`)**: `CLS` 벡터를 입력받아 최종적으로 각 레이블에 대한 로짓(logits)을 계산하는 역할을 합니다. 이는 분류 문제의 출력으로 사용됩니다.

                    ### 요약
                    - `model.py`는 ABSA 작업에서 **카테고리 분류**와 **감성 분류**를 각각 담당하는 모델을 정의합니다.
                    - **사전 학습된 트랜스포머 모델**(XLM-RoBERTa)을 사용하여 문장의 의미를 벡터로 인코딩하고, 이를 **카테고리**와 **감성 레이블**로 변환합니다.
                    - 모델은 입력 문장의 정보를 `CLS` 토큰 벡터에 압축한 뒤, 이를 선형 레이어를 통해 분류 작업을 수행합니다.
                    ''')
    with tab3:
        st.markdown('''
        #### 흐름
        1. **데이터셋 로드 및 DataLoader 설정**: `ABSADataset` 클래스를 사용하여 데이터를 로드하고, PyTorch의 `DataLoader`를 사용해 학습 및 평가 배치를 설정합니다.
        2. **모델 초기화**: `model.py`에서 정의된 `AspectCategoryModel`과 `SentimentModel`을 초기화하고, 학습에 사용할 장치(GPU 또는 CPU)를 설정합니다.
        3. **모델 학습 및 검증**: 모델을 학습시키고, 검증 데이터를 사용해 성능을 평가하는 함수를 정의합니다.
        4. **테스트 데이터 예측 및 저장**: 테스트 데이터를 사용하여 예측을 수행하고, 결과를 JSON 형식으로 저장합니다.
        ''')
        st.markdown('''
        #### 데이터셋 로드 및 DataLoader 설정
        **주요 역할**: 훈련, 검증, 테스트 데이터셋을 `ABSADataset` 클래스를 사용하여 로드하고, `DataLoader`를 사용해 배치 처리를 설정합니다. 이를 통해 데이터가 모델에 효율적으로 공급될 수 있도록 합니다.

        ''')
        st.code('''
                train_dataset = ABSADataset('data/nikluge-sa-2022-train.jsonl', tokenizer)
                val_dataset = ABSADataset('data/nikluge-sa-2022-dev.jsonl', tokenizer)
                test_dataset = ABSADataset('data/nikluge-sa-2022-test.jsonl', tokenizer, is_test=True)

                train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=16)
                test_loader = DataLoader(test_dataset, batch_size=16)

        ''')
        st.markdown('''
        - `ABSADataset` 클래스를 사용해 훈련, 검증, 테스트 데이터를 로드합니다.
        - `DataLoader`는 배치 크기(`batch_size=16`)와 함께 데이터를 모델에 순차적으로 공급할 수 있도록 합니다. 훈련 데이터는 무작위로 섞어(`shuffle=True`) 학습의 일반화 성능을 높입니다.
        ''')
        st.markdown('''
        #### 모델 초기화
        **주요 역할**: `model.py`에서 정의된 모델을 초기화하고, 학습에 사용할 장치를 설정합니다. 사전 학습된 모델을 기반으로 학습을 시작할 수 있게 합니다.

        ''')
        st.code('''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = 'xlm-roberta-base'
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

        num_sentiment_labels = len(train_dataset.sentiment_map)
        num_category_labels = len(train_dataset.category_map)

        category_model = AspectCategoryModel(model_name, num_category_labels).to(device)
        sentiment_model = SentimentModel(model_name, num_sentiment_labels).to(device)
        ''')
        st.markdown('''
        - **모델 및 토크나이저 초기화**: `XLM-RoBERTa`를 사전 학습된 모델로 사용하며, `model.py`에서 정의된 카테고리와 감성 예측 모델을 초기화합니다.
        - 각 모델은 `to(device)`를 통해 GPU나 CPU로 이동하여 계산을 수행할 준비를 합니다.
        ''')
        st.markdown('''
        #### 모델 학습 및 검증
        **주요 역할**: 학습 과정을 정의하고, 모델이 훈련 데이터에 대해 학습하고 검증 데이터로 성능을 평가하도록 합니다. 검증 과정은 모델의 과적합을 방지하고 성능을 모니터링하는 데 사용됩니다.

        ''')
        st.code('''
                def train_model(model, train_loader, val_loader, device, model_type='category'):
                    # 학습 설정: 손실 함수와 옵티마이저
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
                    
                    for epoch in range(num_epochs):
                        model.train()
                        for batch in train_loader:
                            input_ids = batch['input_ids'].to(device)
                            attention_mask = batch['attention_mask'].to(device)
                            labels = batch[model_type].to(device)
                            
                            outputs = model(input_ids, attention_mask)
                            loss = criterion(outputs, labels)
                            
                            # 역전파 및 최적화
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        
                        # 검증 성능 평가
                        model.eval()
                        val_loss = 0
                        with torch.no_grad():
                            for batch in val_loader:
                                input_ids = batch['input_ids'].to(device)
                                attention_mask = batch['attention_mask'].to(device)
                                labels = batch[model_type].to(device)
                                
                                outputs = model(input_ids, attention_mask)
                                loss = criterion(outputs, labels)
                                val_loss += loss.item()
                        
                        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}")
                    return model
        ''')
        st.markdown('''
        - **손실 함수와 옵티마이저 설정**: `CrossEntropyLoss`를 사용하여 분류 작업에 대한 손실을 계산하고, `AdamW` 옵티마이저를 사용해 모델의 가중치를 업데이트합니다.
        - **학습 루프**: 각 에포크마다 훈련 데이터에 대해 손실을 계산하고, 역전파(backpropagation)를 통해 모델을 업데이트합니다.
        - **검증 루프**: 검증 데이터에 대해 손실을 평가하여 모델의 성능을 모니터링합니다. 검증 손실을 통해 모델이 훈련 데이터에 과적합(overfit)되지 않도록 조정할 수 있습니다.
        ''')
        st.markdown('''
        #### 테스트 데이터 예측 및 저장
        **주요 역할**: 학습이 완료된 모델을 사용해 테스트 데이터에 대한 예측을 생성하고, 결과를 JSON 형식으로 저장합니다.
        ''')
        st.code('''
        def predict(model, test_loader, device, model_type='category'):
            model.eval()
            predictions = []
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    outputs = model(input_ids, attention_mask)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    predictions.extend(preds)
            return predictions

        def save_predictions(category_predictions, sentiment_predictions, input_file, output_file, test_dataset):
            # 예측된 카테고리와 감성 결과를 저장합니다.
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
                        print(f"Predictions saved to {output_file}")
        ''')
        st.markdown('''
        - **`predict` 함수**: 테스트 데이터셋에 대해 모델의 예측을 수행합니다. `argmax`를 사용해 각 샘플에 대한 가장 높은 확률을 가진 레이블을 선택합니다.
        - **`save_predictions` 함수**: 예측된 결과를 JSON 형식으로 파일에 저장합니다. 이는 모델의 예측 결과를 확인하고 실제 사용 사례에 활용할 수 있도록 해줍니다.
        ''')    
    with tab4:
        st.markdown('''
        #### 제출 결과
        ''')
        re1 , re2 = st.columns([2,1])
        with re1:
            st.image('./img/base_result.png',width=1000)
        with re2:
            st.markdown('''
### 1. 모델 설명 및 점수 비교
- **klue_base**
  - 성능 점수: `55.1724138`
  
- **xlm-roberta-base**
  - 성능 점수: `53.3333333`

- **KO/electra_base**
  - 성능 점수: `37.5000000`
                        ''')
        re3, re4 = st.columns([1,1])
        with re3:
            st.image('./img/klue_cat.png',width=1000)
        with re4:
            st.markdown('''
            #### Klue 모델의 속성 분류 그래프
            
            ##### 주요 포인트
            - 7,9,10 값인 속성들은 주로 데이터가 많은 속성들입니다. 300개 이상의 데이터가 있으며, 이 그래프를 통해 데이터 증강이 필요하다는 것을 알 수 있습니다.
                - '본품#품질': 7
                - '제품 전체#일반': 9
                - '브랜드#일반': 10
            ''')
        re5, re6 = st.columns([1,1])
        with re5:
            st.image('./img/klue_se.png',width=1000)
        with re6:
            st.markdown('''
            #### Klue 모델의 감성 분류 그래프
            
            ##### 주요 포인트
            - 긍정의 압도적인 분포로 인해, 제대로 학습이 되고 있지 않습니다. **중립** 과 **부정** 의 데이터를 증강이 필요하다는 것을 알 수 있습니다.
                - 0 : 긍정
                - 1 : 중립
                - 2 : 부정
            ''')
        