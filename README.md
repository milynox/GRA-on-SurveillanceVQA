# GRA on Surveillance Video-QA Dataset

Đây là phần source code tham khảo từ bài báo [Video Question Answering via Gradually Refined Attention over Appearance and Motion](https://dl.acm.org/doi/10.1145/3123266.3123427) để chạy trên tập dữ liệu về giám sát an ninh do nhóm xây dựng nên.

Mô hình này chỉ huấn luyện được khi máy có hỗ trợ GPU. Khuyến khích chạy mô hình trên Google Colab nếu máy không có GPU, xem thêm hướng dẫn chạy trên Colab tại [đây](https://colab.research.google.com/drive/1IRPt2vFcdkDG_xgIP_r7Jghwt41dg_fL?usp=sharing)

## Thiết lập
1. Tải code về máy
```
 git clone https://github.com/milynox/GRA-on-SurveillanceVQA.git
```

2. Tải các mô hình đễ hỗ trợ chạy quá trình huấn luyện [Util_Model](https://drive.google.com/file/d/1-ega9COG7bE3s_KQEZeAStKhgbNaArkj/view?usp=sharing). Giải nén và đưa tất cả các file này vào đường dẫn `GRA-on-SurveillanceVQA/util`.

3. Tải tập dữ liệu video giám sát của nhóm [SNN-QA](https://drive.google.com/file/d/1MuEtb_FVnJFfZ33gPI0SLMcxUoXf50NF/view?usp=sharing) về máy, sau đó để các video vào thư mục theo đường dẫn `GRA-on-SurveillanceVQA/video_data`. Những file chứa câu hỏi đã có sẵn trong tương ứng từng thư mục `withUnknownDataset` và `withoutUnknownDataset`.

4. Đổi tên các video để phục vụ tiền xử lý bằng đoạn code sau
```
import os
video_folder_path = r'GRA-on-SurveillanceVQA/video_data'
with open('GRA-on-SurveillanceVQA/video_name_mapping.txt', 'r') as f:
    lines = f.readlines()
    for l in lines:
        l = l.rstrip()
        input_fname = l.split(' ')[0]
        output_fname = l.split(' ')[1]
                
        inputpath = os.path.join(video_folder_path, input_fname)
        outputpath = os.path.join(video_folder_path, output_fname)
        os.rename(inputpath, outputpath)
```


5. Cài đặt các thư viện cần thiết để chạy mô hình:
```bash
!pip install tensorflow-gpu==1.15.0
!pip install sk_video==1.1.8
```
6. Thay đổi thư mục đang làm việc
```
%cd GRA-on-SurveillanceVQA/video_data
```

## Thí nghiệm trên tập dữ liệu hỏi-đáp (qaDataset)
### Tiền xử lý video và câu hỏi
Sao chép video vào thư mục `qaDataset`
```
!mkdir qaDataset/video
!cp -r video_data/* qaDataset/video
```

Sau đó dụng chạy tiền xử lý bằng lệnh:
```
!python preprocess_SQADataset.py qaDataset
```

### Huấn luyện mô hình
Sử dụng lệnh sau để huấn luyện mô hình
Mặc dù dataset có tên là qaDataset nhưng tên thư mục dataset đã tiền xử lý là msvd-qa. Chúng tôi vẫn chưa chuyển đổi code đang định dạng phù hợp, xin thông cảm.
```
!python run_gra.py --mode train --gpu 0 --log log/evqa --dataset msvd_qa --config 0
```

Trong đó các thư mục `log` lưu các checkpoint của mô hình, `data` lưu dữ liệu đã được tiền xử lý.

### Kiểm thử mô hình
Sử dụng lệnh: 
```
!python run_gra.py --mode test --gpu 0 --log log/evqa --dataset msvd_qa --config 0
```
### Lưu kết quả từ mô hình
Kết quả dự đoán là file `log/evqa/prediction.json`. Quá trình huấn luyện và đánh giá lần lượt là các file: `log/evqa/stats/train.json` và `log/evqa/stats/val.json`.
