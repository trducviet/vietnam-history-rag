/**
 * Historical Entity Dictionary — Stage 12C
 *
 * Domain-specific Vietnamese history entity lookup table.
 * All entries are hand-curated from known project entities and corpus metadata.
 * This file DOES NOT mutate any source/corpus/chunk text.
 *
 * Coverage: events, persons, organizations, documents, dates, topics.
 */

import type { HistoricalEntityEntry } from './query-normalization-types.js';

export const HISTORICAL_ENTITY_DICTIONARY: HistoricalEntityEntry[] = [
  // ──────────────────────────────────────────────────────────────
  // EVENTS
  // ──────────────────────────────────────────────────────────────
  {
    canonical: 'Chiến dịch Điện Biên Phủ',
    type: 'event',
    aliases: [
      'dien bien phu', 'dbp', 'chien dich dien bien phu', 'dien bien phu 1954',
      'chien dich dbp', 'tran dien bien phu', 'dbp 1954', 'dien biem phu',
      'dien bien phuu', 'chie dicch dienn bin phu', 'dien bien phu nam 1954',
      'chien dich dbp 1954',
    ],
    no_accent: 'chien dich dien bien phu',
    keywords: ['1954', 'Pháp', 'Genève', 'Võ Nguyên Giáp'],
    confidence_boost: 0.05,
    notes: '13/3/1954 – 7/5/1954',
  },
  {
    canonical: 'Điện Biên Phủ trên không 1972',
    type: 'event',
    aliases: [
      'dien bien phu tren khong', 'dbp tren khong', 'dien bien phu tren khong 1972',
      'chien dich dien bien phu tren khong', 'b52 ha noi', 'rac tham ha noi',
      'dbp 1972', 'dien bien phu khong quan',
    ],
    no_accent: 'dien bien phu tren khong 1972',
    keywords: ['1972', 'B-52', 'Hà Nội', 'không quân'],
    confidence_boost: 0.05,
    notes: '18–29/12/1972, "Điện Biên Phủ trên không"',
  },
  {
    canonical: 'Hiệp định Genève 1954',
    type: 'document',
    aliases: [
      'hiep dinh geneve', 'geneve', 'hiep dinh geneve 1954', 'geneve 1954',
      'hoi nghi geneve', 'hiep din genev', 'hiep dinh genevee', 'hiep dinhh geneve',
      'hiep dinh geneve chia viet nam', 'hiep uoc geneve',
    ],
    no_accent: 'hiep dinh geneve 1954',
    keywords: ['1954', 'vĩ tuyến 17', 'chia cắt', 'Pháp'],
    confidence_boost: 0.03,
    notes: 'Ký ngày 20/7/1954, chia cắt VN tại vĩ tuyến 17',
  },
  {
    canonical: 'Hiệp định Paris 1973',
    type: 'document',
    aliases: [
      'hiep dinh paris', 'paris 1973', 'hiep dinh paris 1973',
      'hoa binh paris', 'hiep uoc paris', 'hiep dinh paris ve viet nam',
      'hiep din paris', 'hiep dinhh paris', 'paris nam 1973',
    ],
    no_accent: 'hiep dinh paris 1973',
    keywords: ['1973', 'Mỹ', 'rút quân', 'hòa bình'],
    confidence_boost: 0.03,
    notes: 'Ký ngày 27/1/1973, Mỹ rút quân khỏi VN',
  },
  {
    canonical: 'Cách mạng Tháng Tám 1945',
    type: 'event',
    aliases: [
      'cach mang thang tam', 'cm thang tam', 'cm thang 8', 'cm t8',
      'cach mang thang 8', 'cach mang t8', 'cach mang thang tam 1945',
      'tong khoi nghia thang tam', 'khoi nghia thang tam',
      'cach mang thag tam', 'cach mang thangg tamm', 'cach mang thang tamm',
      'cuoc cach mang thang tam', 'cuoc khoi nghia thang tam 1945',
      'thang tam nam 1945',
    ],
    no_accent: 'cach mang thang tam 1945',
    keywords: ['1945', 'Việt Minh', 'Hồ Chí Minh', 'khởi nghĩa'],
    confidence_boost: 0.05,
    notes: 'Tháng 8/1945, Việt Minh giành chính quyền',
  },
  {
    canonical: 'Tổng tiến công và nổi dậy Tết Mậu Thân 1968',
    type: 'event',
    aliases: [
      'mau than', 'tet mau than', 'mau than 1968', 'tet mau than 1968',
      'tong tien cong mau than', 'tong tien cong tet mau than',
      'tong tien cong 1968', 'cuoc tong tien cong mau than',
      'mau thn 1968', 'mau thaan 1968', 'tet mau thn',
    ],
    no_accent: 'tong tien cong va noi day tet mau than 1968',
    keywords: ['1968', 'Tết', 'tiến công', 'miền Nam'],
    confidence_boost: 0.05,
    notes: '30–31/1/1968, tấn công đồng loạt miền Nam',
  },
  {
    canonical: 'Chiến dịch Hồ Chí Minh 1975',
    type: 'event',
    aliases: [
      'chien dich ho chi minh', 'chien dich hcm 1975', 'hcm 1975',
      'giai phong sai gon', 'giai phong mien nam', 'chien dich hcm',
      'chien dich ket thuc chien tranh', 'tien cong 30 4',
    ],
    no_accent: 'chien dich ho chi minh 1975',
    keywords: ['1975', 'Sài Gòn', 'giải phóng', 'thống nhất'],
    confidence_boost: 0.05,
    notes: '26–30/4/1975',
  },
  {
    canonical: 'Phong trào Cần Vương',
    type: 'event',
    aliases: [
      'can vuong', 'phong trao can vuong', 'chieu can vuong',
      'phong trao can vuong 1885', 'can vuong 1885',
    ],
    no_accent: 'phong trao can vuong',
    keywords: ['1885', 'Hàm Nghi', 'Tân Sở', 'chống Pháp'],
    confidence_boost: 0.03,
  },
  {
    canonical: 'Phong trào Đông Du',
    type: 'event',
    aliases: [
      'dong du', 'phong trao dong du', 'phan boi chau dong du',
      'dong du 1905', 'dong du nhat ban',
    ],
    no_accent: 'phong trao dong du',
    keywords: ['Phan Bội Châu', 'Nhật Bản', '1905'],
    confidence_boost: 0.03,
  },
  {
    canonical: 'Phong trào Đồng Khởi',
    type: 'event',
    aliases: [
      'dong khoi', 'phong trao dong khoi', 'dong khoi 1960',
      'dong khoi ben tre',
    ],
    no_accent: 'phong trao dong khoi',
    keywords: ['1960', 'Bến Tre', 'miền Nam'],
    confidence_boost: 0.03,
  },
  {
    canonical: 'Xô viết Nghệ Tĩnh',
    type: 'event',
    aliases: [
      'xo viet nghe tinh', 'xo viet nghe an', 'phong trao xo viet nghe tinh',
      'xo viet 1930',
    ],
    no_accent: 'xo viet nghe tinh',
    keywords: ['1930', 'Nghệ An', 'Hà Tĩnh', 'công nhân'],
    confidence_boost: 0.03,
  },
  {
    canonical: 'kháng chiến chống Pháp',
    type: 'topic',
    aliases: [
      'khang chien chong phap', 'chong phap', 'khang chien phap',
      'cuoc khang chien chong phap', 'khang chien 1946 1954',
    ],
    no_accent: 'khang chien chong phap',
    keywords: ['Pháp', 'kháng chiến', '1946', '1954'],
    confidence_boost: 0.02,
  },
  {
    canonical: 'kháng chiến chống Mỹ',
    type: 'topic',
    aliases: [
      'khang chien chong my', 'chong my', 'khang chien my',
      'chien tranh viet nam', 'cuoc chien chong my', 'khang chien 1954 1975',
      'chien tranh viet nam my',
    ],
    no_accent: 'khang chien chong my',
    keywords: ['Mỹ', 'kháng chiến', '1954', '1975'],
    confidence_boost: 0.02,
  },
  {
    canonical: 'hậu phương miền Bắc',
    type: 'topic',
    aliases: [
      'hau phuong mien bac', 'mien bac hau phuong', 'hau phuong',
      'mien bac trong khang chien', 'hau phuong chong my',
    ],
    no_accent: 'hau phuong mien bac',
    keywords: ['miền Bắc', 'hậu phương', 'kháng chiến'],
    confidence_boost: 0.02,
  },
  // ──────────────────────────────────────────────────────────────
  // PERSONS
  // ──────────────────────────────────────────────────────────────
  {
    canonical: 'Nguyễn Ái Quốc',
    type: 'person',
    aliases: [
      'nguyen ai quoc', 'naq', 'NAQ', 'nguyen ai quoc thanh lap dang',
      'nguyen ai quoc lap dang', 'nguen ai quoc', 'nguyen ai quocc',
    ],
    no_accent: 'nguyen ai quoc',
    keywords: ['Hồ Chí Minh', 'Đảng Cộng sản', '1930', 'Pắc Bó'],
    confidence_boost: 0.03,
    notes: 'Bí danh của Hồ Chí Minh trước 1945',
  },
  {
    canonical: 'Hồ Chí Minh',
    type: 'person',
    aliases: [
      'ho chi minh', 'hcm', 'bac ho', 'cu ho', 'nguyen tat thanh',
      'nguyen sinh cung', 'ong ho',
    ],
    no_accent: 'ho chi minh',
    keywords: ['Tuyên ngôn', '1945', 'Cách mạng', 'Chủ tịch'],
    confidence_boost: 0.03,
  },
  {
    canonical: 'Võ Nguyên Giáp',
    type: 'person',
    aliases: [
      'vo nguyen giap', 'tuong giap', 'dai tuong giap',
      'vo nguyen giap dien bien phu',
    ],
    no_accent: 'vo nguyen giap',
    keywords: ['Điện Biên Phủ', 'quân sự', 'Đại tướng'],
    confidence_boost: 0.02,
  },
  {
    canonical: 'Phan Bội Châu',
    type: 'person',
    aliases: [
      'phan boi chau', 'phan boi chau dong du',
    ],
    no_accent: 'phan boi chau',
    keywords: ['Đông Du', 'Nhật Bản'],
    confidence_boost: 0.02,
  },
  // ──────────────────────────────────────────────────────────────
  // ORGANIZATIONS
  // ──────────────────────────────────────────────────────────────
  {
    canonical: 'Việt Minh',
    type: 'organization',
    aliases: [
      'viet minh', 'mat tran viet minh', 'lien minh viet minh',
      'viet minhh', 'viet mnih', 'vietminh', 'viet minh 1941',
    ],
    no_accent: 'viet minh',
    keywords: ['Hồ Chí Minh', '1941', 'Cách mạng Tháng Tám'],
    confidence_boost: 0.03,
  },
  {
    canonical: 'Đảng Cộng sản Việt Nam',
    type: 'organization',
    aliases: [
      'dang cong san viet nam', 'dcsv', 'dcsvu', 'dang cong san', 'dcsvn',
      'nguyen ai quoc thanh lap dang', 'thanh lap dang 1930',
      'dang cong san viet namm', 'dang cs viet nam',
    ],
    no_accent: 'dang cong san viet nam',
    keywords: ['1930', 'Nguyễn Ái Quốc', '3/2/1930'],
    confidence_boost: 0.03,
    notes: 'Thành lập 3/2/1930',
  },
  // ──────────────────────────────────────────────────────────────
  // DOCUMENTS / DATES
  // ──────────────────────────────────────────────────────────────
  {
    canonical: 'Tuyên ngôn Độc lập 2/9/1945',
    type: 'document',
    aliases: [
      'tuyen ngon doc lap', 'tuyen ngon doc lap 2/9/1945', 'tuyen ngon doc lap 1945',
      '2/9/1945', '2/9', 'ngay 2 thang 9 nam 1945', 'doc lap 1945',
      'tuyen ngon 1945', 'tuyen ngon doc laap', 'tuyen ngon',
    ],
    no_accent: 'tuyen ngon doc lap 2/9/1945',
    keywords: ['Hồ Chí Minh', '1945', 'Hà Nội', 'Quảng trường Ba Đình'],
    confidence_boost: 0.05,
    notes: 'Đọc ngày 2/9/1945 tại Quảng trường Ba Đình',
  },
  {
    canonical: '30/4/1975',
    type: 'date',
    aliases: [
      '30/4', '30 thang 4 1975', '30/4/1975', 'ngay 30 thang 4',
      'ngay giai phong', 'ngay 30-4-1975', '30 4 1975',
    ],
    no_accent: '30/4/1975',
    keywords: ['giải phóng', 'Sài Gòn', 'thống nhất', '1975'],
    confidence_boost: 0.05,
    notes: 'Ngày giải phóng miền Nam, thống nhất đất nước',
  },
  {
    canonical: 'thống nhất đất nước 1975',
    type: 'topic',
    aliases: [
      'thong nhat dat nuoc', 'thong nhat 1975', 'thong nhat viet nam',
      'dat nuoc thong nhat', 'giai phong dat nuoc',
    ],
    no_accent: 'thong nhat dat nuoc 1975',
    keywords: ['1975', '30/4', 'Sài Gòn'],
    confidence_boost: 0.02,
  },
  {
    canonical: 'thống nhất nhà nước năm 1976',
    type: 'topic',
    aliases: [
      'thong nhat nha nuoc 1976', 'tong tuyen cu 1976', 'nuoc chxhcnvn 1976',
      'cong hoa xa hoi chu nghia viet nam 1976', 'thong nhat 1976',
    ],
    no_accent: 'thong nhat nha nuoc nam 1976',
    keywords: ['1976', 'Quốc hội', 'tổng tuyển cử'],
    confidence_boost: 0.02,
    notes: 'Tổng tuyển cử 25/4/1976, thành lập nước CHXHCNVN',
  },
  {
    canonical: '3/2/1930',
    type: 'date',
    aliases: [
      '3/2/1930', 'ngay thanh lap dang', 'thanh lap dang 1930',
      'ngay 3 thang 2 1930',
    ],
    no_accent: '3/2/1930',
    keywords: ['Đảng Cộng sản', '1930', 'Nguyễn Ái Quốc'],
    confidence_boost: 0.04,
  },
  {
    canonical: '19/8/1945',
    type: 'date',
    aliases: [
      '19/8/1945', '19/8', 'ngay 19 thang 8', 'ngay 19-8-1945',
      'khoi nghia ha noi', 'ngay khoi nghia ha noi',
    ],
    no_accent: '19/8/1945',
    keywords: ['khởi nghĩa', 'Hà Nội', 'Cách mạng Tháng Tám'],
    confidence_boost: 0.04,
  },
];

/** Build a lookup map from no-accent/alias string → entity entry */
export function buildAliasIndex(
  dict: HistoricalEntityEntry[],
): Map<string, HistoricalEntityEntry> {
  const map = new Map<string, HistoricalEntityEntry>();
  for (const entry of dict) {
    // Index no_accent form
    if (!map.has(entry.no_accent)) map.set(entry.no_accent, entry);
    // Index all aliases
    for (const alias of entry.aliases) {
      if (!map.has(alias)) map.set(alias, entry);
    }
  }
  return map;
}
