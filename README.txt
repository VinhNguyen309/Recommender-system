---Hướng dẫn cài đặt trước khi sử dụng---
+ Cài đặt python và các thư viện cần thiết(ở đây khuyến khích sử dụng Anaconda- tích hợp sẵn các thư viện cần thiết)
+Cài đặt nodejs 
---Chạy demo---
Vì đây là dạng demo cho việc chạy trên serve thực nên cần khởi động 2 serve trên localhost để có thể chạy được REST API
---Khởi động serve thuật toán---
Đầu tiên mở cmd command, chạy file "Algorithm_Recommend.py" trong cùng thư mục để chạy được serve của thuật toán trả về
---Khởi động serve web---
Tiếp theo mở cmd command , chạy câu lênh "npm install -g @angular/cli " để cài đặt tất cả các package cần thiết có trong project
sau đó chạy tiếp lệnh "ng serve --open" để khởi động serve của web
Sau khi chạy xong cần tắt web đó đi, vì một số lý do nên cần xử lý về trình duyệt để có thể chạy được.
Mở icon của trình duyệt vào properties -> shortcut. Tìm mục target và thêm " --disable-web-security --user-data-dir" đoạn lệnh 
trên vào cuối chuỗi. Sau đó nhấn ok. Vậy là xong ^^!!!!

Good luck! 