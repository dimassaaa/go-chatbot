package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// Content merepresentasikan isi pesan (Parts) beserta peran pengirimnya (Role),
// digunakan untuk memparsing struktur respons dari Gemini API.
type Content struct {
	Parts []string `json:"Parts"`
	Role  string   `json:"Role"`
}

// Candidates membungkus satu kandidat jawaban yang dikembalikan oleh model,
// setiap kandidat memiliki Content-nya sendiri.
type Candidates struct {
	Content *Content `json:"Content"`
}

// ContentResponse adalah struktur hasil parsing dari response mentah Gemini
// (resp) setelah di-marshal ulang ke JSON lalu di-unmarshal ke bentuk custom ini.
type ContentResponse struct {
	Candidates *[]Candidates `json:"Candidates"`
}

// userPrompt adalah body request yang dikirim client ke endpoint /gemini,
// berisi teks pertanyaan/prompt dari pengguna.
type userPrompt struct {
	Prompt string `json:"prompt"`
}

// result adalah struktur response yang dikirim balik ke client,
// berisi jawaban akhir dari model AI.
type result struct {
	Cresult string `json:"result"`
}

func main() {
	// Inisialisasi router Gin dan aktifkan CORS default
	// agar API bisa diakses dari domain frontend yang berbeda.
	router := gin.Default()
	router.Use(cors.Default())

	ctx := context.Background()

	// Membuat client Gemini menggunakan API Key dari environment variable.
	// Jika API Key tidak valid/tidak ada, aplikasi langsung dihentikan (log.Fatal).
	client, err := genai.NewClient(ctx, option.WithAPIKey(os.Getenv("API_KEY")))
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	// Menentukan model Gemini yang digunakan.
	model := client.GenerativeModel("gemini-3.6-flash")

	// System instruction: mendefinisikan persona AI sebagai asisten
	// Lucky Rental, membatasi topik jawaban hanya seputar informasi
	// perusahaan yang diberikan, dan mewajibkan jawaban berbahasa Indonesia.
	Instruction := "You are a personal assistant for the lucky rental website. Fluent and only speaks Bahasa Indonesia. I provide you this information: Tentang Perusahaann Nama Perusahaan: Lucky Rental Tahun Berdiri: 2024 Lokasi Kantor Pusat: Malang, Indonesia Layanan yang Ditawarkan Penyewaan mobil untuk perorangan dan perusahaan. Berbagai jenis mobil: sedan, SUV, minivan. Mobil-mobil dalam kondisi terawat dan bersih.  Layanan antar jemput dari dan ke bandara. Fitur Website Pemesanan online mudah dan cepat. Pilihan mobil dengan spesifikasi detail. Penawaran khusus dan diskon untuk pelanggan setia. Review dan testimoni pelanggan. Proses Pemesanan Pelanggan dapat memilih jenis mobil, tanggal, dan lokasi pengambilan mobil. Konfirmasi pemesanan via email. Opsi pembayaran melalui kartu kredit atau transfer bank.  Ketersediaan Layanan Dapat diakses 24/7. Dukungan pelanggan melalui live chat dan telepon. Informasi Tambahan Syarat dan ketentuan sewa: Penyewa harus memiliki KTP dan SIM agar bisa menyewa kendaraan di Lucky Rental Kebijakan penyewaan: Jika ada kerusakan pada kendaraan sewa akan ditanggung oleh penyewa Kontak Perusahaan Nomor telepon: 082139020016 Email: luckyrental@gmail.com You can't answer questions out of those information. If you asked about full information of this website dont share all of what i give to you, but modify it to humanly language"

	// Membungkus instruction di atas ke dalam genai.Content
	// lalu menetapkannya sebagai SystemInstruction model.
	instructionContent := &genai.Content{
		Parts: []genai.Part{
			genai.Text(Instruction),
		},
	}
	model.SystemInstruction = instructionContent

	// Menonaktifkan filter safety untuk kategori konten berbahaya (dangerous content)
	model.SafetySettings = []*genai.SafetySetting{
		{
			Category:  genai.HarmCategoryDangerousContent,
			Threshold: genai.HarmBlockNone,
		},
		{
			Category:  genai.HarmCategoryHarassment,
			Threshold: genai.HarmBlockNone,
		},
	}

	// Endpoint utama: menerima prompt dari client, mengirimkannya ke Gemini,
	// lalu mengembalikan jawaban model sebagai response JSON.
	router.POST("/gemini", func(c *gin.Context) {
		// Bind body request JSON ke struct userPrompt.
		var inpPrompt userPrompt
		if err := c.BindJSON(&inpPrompt); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		// Mengirim prompt ke Gemini API dan menunggu hasil generate content.
		resp, err := model.GenerateContent(ctx, genai.Text(inpPrompt.Prompt))
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		// Response asli dari SDK genai memiliki struktur yang kompleks,
		// sehingga di-marshal ke JSON lalu di-unmarshal ulang ke struct
		// custom (ContentResponse) agar lebih mudah diproses.
		marshalResponse, _ := json.MarshalIndent(resp, "", "  ")
		var generateResponse ContentResponse
		if err := json.Unmarshal(marshalResponse, &generateResponse); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}

		// Mengambil teks jawaban dari setiap kandidat & bagian (Parts),
		// lalu mengirimkannya sebagai response ke client.
		for _, cad := range *generateResponse.Candidates {
			if cad.Content != nil {
				for _, part := range cad.Content.Parts {
					c.JSON(http.StatusOK, result{part})
				}
			}
		}
	})

	// Menjalankan server di port 8080.
	router.Run(":8080")
}
