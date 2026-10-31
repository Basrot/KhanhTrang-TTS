# Khánh Trang Reader v0.7 — sửa lỗi nút Play

Bản trước đưa URL `http://127.0.0.1:8765/api/audio/...` trực tiếp vào thẻ `<audio>` trên trang báo. Một số trang có chính sách bảo mật chặn phát media từ localhost, làm nút Play không hoạt động.

Bản này tải WAV qua service worker của extension (có quyền truy cập localhost), truyền từng phần vào giao diện, tạo Blob URL và dùng trình phát HTML5. Nút Play chỉ sẵn sàng khi tải xong âm thanh.

Cập nhật: giải nén ZIP, chép đè `khanhtrang_extension` vào `C:\Khanhtrang-tts`; khởi động lại backend với UTF-8; Reload extension tại edge://extensions và F5 trang báo.

Lưu ý: file WAV lớn có thể cần thời gian tải và dùng RAM. Đây là bản sửa chưa kiểm thử trực tiếp trên Edge máy bạn.
