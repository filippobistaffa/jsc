switch (f->c) {
	case 1: { NATIVESORT(f, I); break; }
	case 2: { templatesort<chunk2,I>(f); break; }
	case 3: { templatesort<chunk3,I>(f); break; }
	case 4: { templatesort<chunk4,I>(f); break; }
	case 5: { templatesort<chunk5,I>(f); break; }
	case 6: { templatesort<chunk6,I>(f); break; }
	case 7: { templatesort<chunk7,I>(f); break; }
	case 8: { templatesort<chunk8,I>(f); break; }
	case 9: { templatesort<chunk9,I>(f); break; }
	case 10: { templatesort<chunk10,I>(f); break; }
	default:;
}
